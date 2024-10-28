"""
Implementation of predictor modules and wrapper functionalities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from sold.utils.model_blocks import PositionalEncoding


class PredictorWrapper(nn.Module):
    """
    Wrapper module that autoregressively applies any predictor module on a sequence of data

    Args:
    -----
    predictor: nn.Module
        Instanciated predictor module to wrap.
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

    def forward(self, slot_history, condition=None):
        """
        Iterating over a sequence of slots, predicting the subsequent slots

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)
        condition: torch Tensor
            One condition for each frame on basis of which the predictor should make its prediction (optional).
            Shape is (B, num_frames, cond_dim)

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        self._is_teacher_force()
        pred_slots = self.forward_action_transformer(slot_history, condition)
        return pred_slots

    def forward_action_transformer(self, slot_history, actions):
        """
        Forward pass through Transformer-based predictor module conditioned on actions

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots form the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)
        condition: torch Tensor
            One action for each frame on basis of which the predictor should make its prediction.
            Shape is (B, num_frames, action_dim)

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        first_slot_idx = 1 if self.skip_first_slot else 0
        predictor_input = slot_history[:, first_slot_idx:self.num_context].clone()  # initial token buffer

        print("slot_history.shape:", slot_history.shape)
        print("actions.shape:", actions.shape)

        input()

        pred_slots = []
        for t in range(self.num_preds):
            print("actions[:, :self.num_context-1+t].clone().shape:", actions[:, :self.num_context-1+t].clone())
            input()
            cur_preds = self.predictor(predictor_input, actions[:, :self.num_context-1+t].clone())[:, -1]  # get predicted slots from step
            next_input = slot_history[:, self.num_context+t] if self.teacher_force else cur_preds
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)
        return pred_slots

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
            cur_preds = self.predictor(predictor_input, input_actions)[:, -1]  # get predicted slots from step
            next_input = cur_preds
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_preds)

        return torch.stack(pred_slots, dim=1)


class DynamicsModel(nn.Module):
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
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output[:, :, :-1])
        output = output + slots if self.residual else output
        return output


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

