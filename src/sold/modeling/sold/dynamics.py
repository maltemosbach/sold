from sold.modeling.positional_encoding import SinusoidalPositionalEncoding, TokenWiseSinusoidalPositionalEncoding
import torch
import torch.nn as nn


class AutoregressiveWrapper(nn.Module):
    def __init__(self, predictor, teacher_forcing: bool):
        super().__init__()
        self.predictor = predictor
        self.input_buffer_size = predictor.input_buffer_size
        self.teacher_forcing = teacher_forcing
        self.batched_processing = True

    def _update_buffer_size(self, inputs):
        num_inputs = inputs.shape[1]
        if num_inputs > self.input_buffer_size:
            extra_inputs = num_inputs - self.input_buffer_size
            inputs = inputs[:, extra_inputs:]
        return inputs

    def predict_slots(self, slots: torch.Tensor, actions: torch.Tensor, steps: int, num_context: int) -> torch.Tensor:
        if self.teacher_forcing and self.batched_processing:
            input_slots = self._update_buffer_size(slots[:, :num_context + steps - 1].clone())
            input_actions = self._update_buffer_size(actions.clone()[:, :num_context + steps - 1])
            predicted_slots = self.predictor(input_slots, input_actions)[:, num_context - 1:]

        else:
            input_slots = self._update_buffer_size(slots[:, :num_context].clone())
            predicted_slots = []
            for t in range(num_context, num_context + steps):
                input_actions = self._update_buffer_size(actions.clone()[:, :t])
                current_predicted_slots = self.predictor(input_slots, input_actions)[:, -1]

                if t < num_context + steps - 1:
                    next_input = slots[:, t] if self.teacher_forcing else current_predicted_slots
                    input_slots = torch.cat([input_slots, next_input.unsqueeze(1)], dim=1)
                    input_slots = self._update_buffer_size(input_slots)
                predicted_slots.append(current_predicted_slots)
            predicted_slots = torch.stack(predicted_slots, dim=1)

        return predicted_slots


class OCVPSeqDynamicsModel(nn.Module):
    def __init__(self, num_slots: int, slot_dim: int, sequence_length: int, action_dim: int, token_dim=128, hidden_dim=256, num_layers=2,
                 num_heads=4, residual=True, input_buffer_size=5):
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

        causal_mask = torch.tril(torch.ones(num_imgs, num_imgs, device=inputs.device)) #.unsqueeze(0).repeat(B * num_slots, 1, 1)
        #print("object_encoded_out", object_encoded_out.shape)
        #print("causal_mask", causal_mask.shape)
        #input()
        object_encoded_out = self.time_encoder_block(object_encoded_out, src_mask=causal_mask, is_causal=True)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out


def make_ocvp_seq_dynamics_model(num_slots: int, slot_dim: int, sequence_length: int, action_dim: int, token_dim=128, hidden_dim=256, num_layers=2,
                 num_heads=4, residual=True, input_buffer_size=5, teacher_forcing=False):
    return AutoregressiveWrapper(OCVPSeqDynamicsModel(num_slots, slot_dim, sequence_length, action_dim, token_dim, hidden_dim, num_layers, num_heads, residual, input_buffer_size), teacher_forcing)

