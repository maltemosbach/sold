import torch
import torch.nn as nn
from sold.modeling.blocks import TransformerBlock


class AutoregressiveWrapper(nn.Module):

    def __init__(self, predictor: "MalteDynamicsModel"):
        """
        Module initializer
        """
        super().__init__()
        self.predictor = predictor

    def predict_slots(self, steps: int, context_slots: torch.Tensor, actions: torch.Tensor):
        context_slots = context_slots.detach().clone()
        actions = actions.detach().clone()

        # Used to update the hidden state of the predictor
        hidden = None
        for context_step in range(context_slots.shape[1] - 1):
            current_slots = context_slots[:, context_step].detach().clone()
            current_actions = actions[:, context_step].detach().clone()
            next_hidden, next_slots = self.predictor(current_slots, current_actions, hidden)
            hidden = next_hidden

        # Used to predict future slots for 'steps' steps.
        predicted_slots = []
        for prediction_step in range(context_slots.shape[1], context_slots.shape[1] + steps):
            current_slots = context_slots[:, prediction_step - 1].detach().clone()
            current_actions = actions[:, prediction_step - 1].detach().clone()
            next_hidden, next_slots = self.predictor(current_slots, current_actions, hidden)
            context_slots = torch.cat([context_slots, next_slots.unsqueeze(1)], dim=1) # Removed detach here
            hidden = next_hidden
            predicted_slots.append(next_slots)
        return torch.stack(predicted_slots, dim=1)


class GRUSAViDynamicsModel(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, sequence_length: int, action_dim: int, token_dim=128, hidden_dim=256, num_layers=2,
                 num_heads=4, residual=True):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.sequence_length = sequence_length
        self.action_dim = action_dim

        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.residual = residual

        # GRUCell to integrate temporal information about slots.
        self.gru = nn.GRUCell(slot_dim, hidden_dim)

        # Transformer to integrate information about slots and actions.
        self.transformer = TransformerBlock(token_dim, num_heads, mlp_size=256)
        self.actions_to_tokens = nn.Linear(self.action_dim, token_dim)
        self.hiddens_to_tokens = nn.Linear(self.hidden_dim, token_dim)
        self.tokens_to_slots = nn.Linear(token_dim, slot_dim)

    def forward(self, slots: torch.Tensor, actions: torch.Tensor, hidden: torch.Tensor = None):
        """
        Forward pass through CondOCVP-Seq

        Args:
            slots (torch.Tensor): Object slots from the previous time step. Shape is (batch_size num_slots, slot_dim).
            actions (torch Tensor): Actions performed by the agent. Shape is (batch_size, action_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        batch_size, num_slots, slot_dim = slots.shape
        if hidden is None:
            hidden = torch.zeros(batch_size * num_slots, self.hidden_dim, device=slots.device)

        # Integrate temporal information of the new set of slots.
        next_hidden = self.gru(
            slots.reshape(-1, slot_dim),
            hidden,
        )

        # On the updated hidden states and embedded action, do self-attention to predict the next set of slots.
        action_tokens = self.actions_to_tokens(actions)
        slot_tokens = self.hiddens_to_tokens(next_hidden).reshape(batch_size, num_slots, self.token_dim)
        tokens = torch.cat((slot_tokens, action_tokens.unsqueeze(1)), dim=1)  # (batch_size, num_slots + 1, token_dim)
        next_tokens = self.transformer(tokens)

        next_slots = self.tokens_to_slots(next_tokens[:, :-1])  # Remove the action token.

        if self.residual:
            next_slots = next_slots + slots.detach()
        return next_hidden, next_slots


def make_gru_savi_dynamics_model(num_slots: int, slot_dim: int, sequence_length: int, action_dim: int, token_dim=128, hidden_dim=256, num_layers=4,
                 num_heads=4, residual=True, input_buffer_size: int = None):
    return AutoregressiveWrapper(GRUSAViDynamicsModel(num_slots, slot_dim, sequence_length, action_dim, token_dim, hidden_dim, num_layers, num_heads, residual))


from sold.modeling.positional_encoding import SinusoidalPositionalEncoding


class TransformerSAViDynamicsModelAutoregressiveWrapper(nn.Module):
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
        context_slots = self._update_buffer_size(slot_history.clone())

        predicted_slots = []
        for prediction_step in range(context_slots.shape[1], context_slots.shape[1] + steps):
            #current_slots = context_slots[:, prediction_step - 1].detach().clone()
            current_actions = actions[:, prediction_step - 1].detach().clone()
            slot_tokens, next_slots = self.predictor(context_slots, current_actions)
            context_slots = torch.cat([context_slots, next_slots.unsqueeze(1)], dim=1)  # Removed detach here
            predicted_slots.append(next_slots)

        return torch.stack(predicted_slots, dim=1)


class TransformerSAViDynamicsModel(nn.Module):
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
        self.residual = False
        self.input_buffer_size = input_buffer_size

        # Linear layers to map from slot_dim to token_dim, and back
        self.slots_to_tokens = nn.Linear(slot_dim, token_dim)
        self.tokens_to_slots = nn.Linear(token_dim, slot_dim)

        # Embed_dim will be split across num_heads, i.e., each head will have dim. embed_dim // num_heads
        self.temporal_transformer = nn.Sequential(
            *[TimeEncoderLayer(
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    n_heads=num_heads
                ) for _ in range(num_layers)]
            )

        self.interaction_transformer = nn.Sequential(
            *[torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=num_heads,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
                ) for _ in range(3)]
        )

        # custom temporal encoding. All slots from the same time step share the same encoding
        self.pe = SinusoidalPositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)
        # Token embedding for action
        self.actions_to_tokens = nn.Linear(self.action_dim, token_dim)

        self.autoregressive_prediction = True

    def forward(self, slots_sequence: torch.Tensor, actions: torch.Tensor):
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
        B, sequence_length, num_slots, slot_dim = slots_sequence.shape

        B, action_dim = actions.shape

        # projecting slots into tokens, and applying positional encoding
        slot_tokens = self.slots_to_tokens(slots_sequence)

        slot_tokens = self.pe(
                x=slot_tokens,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through transformer blocks
        for encoder in self.temporal_transformer:
            slot_tokens = encoder(slot_tokens)

        action_tokens = self.actions_to_tokens(actions)

        tokens = torch.cat((action_tokens.unsqueeze(1), slot_tokens[:, -1]), dim=1)  # (batch_size, num_slots + 1, token_dim)

        if self.autoregressive_prediction:
            next_tokens = []
            for _ in range(num_slots):
                predicted_tokens = self.interaction_transformer(tokens)[:, -1].unsqueeze(1)
                next_tokens.append(predicted_tokens)
                tokens = torch.cat((tokens, predicted_tokens), dim=1)
            next_tokens = torch.cat(next_tokens, dim=1)
        else:
            next_tokens = self.interaction_transformer(tokens)[:, 1:]  # Remove the action token.
        next_slots = self.tokens_to_slots(next_tokens)

        if self.residual:
            next_slots = next_slots + slots_sequence[:, -1].detach()
        return slot_tokens, next_slots


class TimeEncoderLayer(nn.Module):

    def __init__(self, token_dim=128, hidden_dim=256, n_heads=4):
        """
        Module initializer
        """
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.nhead = n_heads

        self.time_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        return

    def forward(self, inputs, time_mask=None):
        B, num_imgs, num_slots, dim = inputs.shape

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        object_encoded_out = inputs.transpose(1, 2)
        object_encoded_out = object_encoded_out.reshape(B * num_slots, num_imgs, dim)
        object_encoded_out = self.time_encoder_block(object_encoded_out)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out


def make_transformer_savi_dynamics_model(num_slots: int, slot_dim: int, sequence_length: int, action_dim: int, token_dim=128, hidden_dim=256, num_layers=4,
                 num_heads=4, residual=True, input_buffer_size: int = None):
    return TransformerSAViDynamicsModelAutoregressiveWrapper(TransformerSAViDynamicsModel(num_slots, slot_dim, sequence_length, action_dim, token_dim, hidden_dim, num_layers, num_heads, residual, input_buffer_size))
