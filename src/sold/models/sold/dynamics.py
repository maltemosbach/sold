from sold.utils.positional_encoding import SinusoidalPositionalEncoding
import torch
import torch.nn as nn


class AutoregressiveWrapper(nn.Module):
    """
    Wrapper module that autoregressively applies any predictor module on a sequence of data

    Args:
    -----
    predictor: nn.Module
        Instantiated predictor module to wrap.
    """

    def __init__(self, predictor):
        """
        Module initializer
        """
        super().__init__()
        self.predictor = predictor

        # prediction parameters
        self.num_context = 1
        self.num_preds = 15
        self.teacher_force = False
        self.skip_first_slot = False
        self.video_length = 16
        self.input_buffer_size = predictor.input_buffer_size

    def _is_teacher_force(self):
        """
        Updating the teacher force value, depending on the training stage
            - In eval-mode, then teacher-forcing is always false
            - In train-mode, then teacher-forcing depends on the predictor parameters
        """
        if self.predictor.train is False:
            self.teacher_force = False
        else:
            self.teacher_force = False
        return

    def _update_buffer_size(self, inputs):
        """
        Updating the inputs of a transformer model given the 'buffer_size'.
        We keep a moving window over the input tokens, dropping the oldest slots if the buffer
        size is exceeded.
        """
        num_inputs = inputs.shape[1]
        if num_inputs > self.input_buffer_size:
            extra_inputs = num_inputs - self.input_buffer_size
            inputs = inputs[:, extra_inputs:]
        return inputs

    def predict_slots(self, steps, slot_history, actions):
        predictor_input = self._update_buffer_size(slot_history.clone())

        pred_slots = []
        for t in range(predictor_input.shape[1], predictor_input.shape[1] + steps):

            input_actions = self._update_buffer_size(actions.clone()[:, :t])
            #print("input_actions.shape:", input_actions.shape)
            token_output, output = self.predictor(predictor_input, input_actions)  # get predicted slots from step
            #print("token_output.shape:", token_output.shape)
            #print("output.shape:", output.shape)
            token_output = token_output[:, -1]
            output = output[:, -1]
            #input()
            next_input = output
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(output)
        return torch.stack(pred_slots, dim=1)




class TokenWiseAutoregressiveWrapper(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

        # prediction parameters
        self.num_context = 1
        self.num_preds = 15
        self.teacher_force = False
        self.skip_first_slot = False
        self.video_length = 16
        self.input_buffer_size = predictor.input_buffer_size

    def _is_teacher_force(self):
        """
        Updating the teacher force value, depending on the training stage
            - In eval-mode, then teacher-forcing is always false
            - In train-mode, then teacher-forcing depends on the predictor parameters
        """
        if self.predictor.train is False:
            self.teacher_force = False
        else:
            self.teacher_force = False
        return

    def _update_buffer_size(self, inputs):
        """
        Updating the inputs of a transformer model given the 'buffer_size'.
        We keep a moving window over the input tokens, dropping the oldest slots if the buffer
        size is exceeded.
        """
        num_inputs = inputs.shape[1]
        if num_inputs > self.input_buffer_size:
            extra_inputs = num_inputs - self.input_buffer_size
            inputs = inputs[:, extra_inputs:]
        return inputs

    def predict_slots(self, steps, slot_history, actions):
        predictor_input = self._update_buffer_size(slot_history.clone())

        batch_size, context_length, num_slots, slot_dim = slot_history.shape

        print("predictor_input.shape:", predictor_input.shape)

        visible_slots = predictor_input.reshape(batch_size, -1, slot_dim)

        visible_tokens = self.predictor.mlp_in(visible_slots)

        pred_slots = []
        for t in range(predictor_input.shape[1], predictor_input.shape[1] + steps):
            #input_actions = self._update_buffer_size(actions.clone()[:, :t])
            embedded_input_actions = self.predictor.action_encoder(actions.clone()[:, t])

            print("visible_tokens.shape:", visible_tokens.shape)
            print("embedded_input_actions.shape:", embedded_input_actions.shape)

            visible_slots_actions = torch.cat([visible_tokens, embedded_input_actions.unsqueeze(1)], dim=1)

            print("visible_slots_actions.shape:", visible_slots_actions.shape)

            for slot_index in range(num_slots):
                next_token_output, next_slot = self.predictor(visible_slots_actions)

                next_slot_token = self.mlp_in(next_slot)
                visible_slots_actions = torch.cat([visible_slots, next_slot_token.unsqueeze(1)], dim=1)

                print("visible_slots_actions.shape:", visible_slots_actions.shape)
                input()


            #print("input_actions.shape:", input_actions.shape)
            token_output, output = self.predictor(predictor_input, input_actions)  # get predicted slots from step
            #print("token_output.shape:", token_output.shape)
            #print("output.shape:", output.shape)
            token_output = token_output[:, -1]
            output = output[:, -1]
            #input()
            next_input = output
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(output)

        return torch.stack(pred_slots, dim=1)


class TokenWiseVanillaTransformerDynamicsModel(nn.Module):
    """
    Conditional Transformer Predictor module.
    In addition, this one gets a condition, e.g., action performed by an agent, for its prediction.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs + 1 (action)
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs + 1 (action)
    cond_dim: int
        Dimensionality of condition input.
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots: int, slot_dim: int, sequence_length: int, action_dim: int, token_dim=128, hidden_dim=256, num_layers=2,
                 num_heads=4, residual=True, input_buffer_size=5):
        """
        Module Initialzer
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.sequence_length = sequence_length
        self.action_dim = action_dim

        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print("Instanciating OCVP-Seq Predictor Module:")
        print(f"  --> num_layers: {self.num_layers}")
        print(f"  --> input_dim: {self.slot_dim}")
        print(f"  --> token_dim: {self.token_dim}")
        print(f"  --> hidden_dim: {self.hidden_dim}")
        print(f"  --> num_heads: {num_heads}")
        print(f"  --> residual: {self.residual}")
        print(f"  --> action_dim: {self.action_dim}")
        print("  --> batch_first: True")
        print("  --> norm_first: True")
        print(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # Embed_dim will be split across num_heads, i.e., each head will have dim. embed_dim // num_heads
        # self.transformer_encoders = nn.Sequential(
        #     *[OCVPSeqLayer(
        #             token_dim=token_dim,
        #             hidden_dim=hidden_dim,
        #             n_heads=num_heads
        #         ) for _ in range(num_layers)]
        #     )

        self.transformer_encoders = nn.Sequential(
            *[torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            ) for _ in range(num_layers)]
        )

        # custom temporal encoding. All slots from the same time step share the same encoding
        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)
        # Token embedding for action
        self.action_encoder = nn.Linear(self.action_dim, token_dim)
        return

    def forward(self, slot_actions):
        """
        Forward pass through CondOCVP-Seq

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)
        action: torch Tensor
            Condition the transformer output should be conditioned on, e.g., action performed by an agent. Shape is (B, cond_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_visible_slots, emb_dim = slot_actions.shape
        print("slot_actions.shape:", slot_actions.shape)
        input()

        # projecting slots into tokens, and applying positional encoding
        token_input = self.mlp_in(slots)
        token_input = torch.cat((token_input, action_embeddings.unsqueeze(2)), dim=2)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots + 1
            )

        # feeding through transformer blocks
        token_output = time_encoded_input.reshape(B, num_imgs * (num_slots + 1), self.token_dim)
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        token_output = token_output.reshape(B, num_imgs, (num_slots + 1), self.token_dim)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output[:, :, :-1])  # Remove action token
        output = output + slots if self.residual else output
        return token_output, output


class VanillaTransformerDynamicsModel(nn.Module):
    """
    Conditional Transformer Predictor module.
    In addition, this one gets a condition, e.g., action performed by an agent, for its prediction.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs + 1 (action)
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs + 1 (action)
    cond_dim: int
        Dimensionality of condition input.
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots: int, slot_dim: int, sequence_length: int, action_dim: int, token_dim=128, hidden_dim=256, num_layers=2,
                 num_heads=4, residual=True, input_buffer_size=5):
        """
        Module Initialzer
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.sequence_length = sequence_length
        self.action_dim = action_dim

        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print("Instanciating OCVP-Seq Predictor Module:")
        print(f"  --> num_layers: {self.num_layers}")
        print(f"  --> input_dim: {self.slot_dim}")
        print(f"  --> token_dim: {self.token_dim}")
        print(f"  --> hidden_dim: {self.hidden_dim}")
        print(f"  --> num_heads: {num_heads}")
        print(f"  --> residual: {self.residual}")
        print(f"  --> action_dim: {self.action_dim}")
        print("  --> batch_first: True")
        print("  --> norm_first: True")
        print(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # Embed_dim will be split across num_heads, i.e., each head will have dim. embed_dim // num_heads
        # self.transformer_encoders = nn.Sequential(
        #     *[OCVPSeqLayer(
        #             token_dim=token_dim,
        #             hidden_dim=hidden_dim,
        #             n_heads=num_heads
        #         ) for _ in range(num_layers)]
        #     )

        self.transformer_encoders = nn.Sequential(
            *[torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            ) for _ in range(num_layers)]
        )

        # custom temporal encoding. All slots from the same time step share the same encoding
        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)
        # Token embedding for action
        self.action_encoder = nn.Linear(self.action_dim, token_dim)
        return

    def forward(self, slots, actions):
        """
        Forward pass through CondOCVP-Seq

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)
        action: torch Tensor
            Condition the transformer output should be conditioned on, e.g., action performed by an agent. Shape is (B, cond_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, slot_dim = slots.shape

        action_embeddings = self.action_encoder(actions)

        # projecting slots into tokens, and applying positional encoding
        token_input = self.mlp_in(slots)
        token_input = torch.cat((token_input, action_embeddings.unsqueeze(2)), dim=2)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots + 1
            )

        # feeding through transformer blocks
        token_output = time_encoded_input.reshape(B, num_imgs * (num_slots + 1), self.token_dim)
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        token_output = token_output.reshape(B, num_imgs, (num_slots + 1), self.token_dim)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output[:, :, :-1])  # Remove action token
        output = output + slots if self.residual else output
        return token_output, output


class OCVPSeqDynamicsModel(nn.Module):
    """
    Conditional Transformer Predictor module.
    In addition, this one gets a condition, e.g., action performed by an agent, for its prediction.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs + 1 (action)
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs + 1 (action)
    cond_dim: int
        Dimensionality of condition input.
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots: int, slot_dim: int, sequence_length: int, action_dim: int, token_dim=128, hidden_dim=256, num_layers=2,
                 num_heads=4, residual=True, input_buffer_size=5):
        """
        Module Initialzer
        """
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.sequence_length = sequence_length
        self.action_dim = action_dim

        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        print("Instanciating OCVP-Seq Predictor Module:")
        print(f"  --> num_layers: {self.num_layers}")
        print(f"  --> input_dim: {self.slot_dim}")
        print(f"  --> token_dim: {self.token_dim}")
        print(f"  --> hidden_dim: {self.hidden_dim}")
        print(f"  --> num_heads: {num_heads}")
        print(f"  --> residual: {self.residual}")
        print(f"  --> action_dim: {self.action_dim}")
        print("  --> batch_first: True")
        print("  --> norm_first: True")
        print(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # Embed_dim will be split across num_heads, i.e., each head will have dim. embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[OCVPSeqLayer(
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    n_heads=num_heads
                ) for _ in range(num_layers)]
            )

        # custom temporal encoding. All slots from the same time step share the same encoding
        self.pe = SinusoidalPositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)
        # Token embedding for action
        self.action_encoder = nn.Linear(self.action_dim, token_dim)
        return

    def forward(self, slots, actions):
        """
        Forward pass through CondOCVP-Seq

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)
        action: torch Tensor
            Condition the transformer output should be conditioned on, e.g., action performed by an agent. Shape is (B, cond_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, slot_dim = slots.shape

        action_embeddings = self.action_encoder(actions)

        # projecting slots into tokens, and applying positional encoding
        token_input = self.mlp_in(slots)
        token_input = torch.cat((token_input, action_embeddings.unsqueeze(2)), dim=2)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots + 1
            )

        # feeding through transformer blocks
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output[:, :, :-1])
        output = output + slots if self.residual else output
        return token_output, output


class OCVPSeqLayer(nn.Module):
    """
    Sequential Object-Centric Video Prediction (OCVP-Seq) Transformer Layer.
    Sequentially applies object- and time-attention.

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    hidden_dim: int
        Hidden dimensionality of the MLPs in the transformer modules
    n_heads: int
        Number of heads for multi-head self-attention.
    """

    def __init__(self, token_dim=128, hidden_dim=256, n_heads=4):
        """
        Module initializer
        """
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.nhead = n_heads

        self.object_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        self.time_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        return

    def forward(self, inputs, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-V1 Layer

        Args:
        -----
        inputs: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        """
        B, num_imgs, num_slots, dim = inputs.shape

        # object-attention block. Operates on (B * N_imgs, N_slots, Dim)
        inputs = inputs.reshape(B * num_imgs, num_slots, dim)
        object_encoded_out = self.object_encoder_block(inputs)
        object_encoded_out = object_encoded_out.reshape(B, num_imgs, num_slots, dim)

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        object_encoded_out = object_encoded_out.reshape(B * num_slots, num_imgs, dim)
        object_encoded_out = self.time_encoder_block(object_encoded_out)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out

