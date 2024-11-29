from sold.modeling.positional_encoding import SinusoidalPositionalEncoding, TokenWiseSinusoidalPositionalEncoding
import torch
import torch.nn as nn
from sold.modeling.blocks import TransformerBlock


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
            cur_preds = self.predictor(predictor_input, input_actions)[:, -1]  # get predicted slots from step
            next_input = cur_preds
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_preds)

        return torch.stack(pred_slots, dim=1)


class MalteDynamicsModel(nn.Module):
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
        self.prev_hidden = None

        # Transformer to integrate information about slots and actions.
        self.transformer = TransformerBlock(token_dim, num_heads, mlp_size=256)
        self.actions_to_tokens = nn.Linear(self.action_dim, token_dim)
        self.hiddens_to_tokens = nn.Linear(self.hidden_dim, token_dim)
        self.tokens_to_slots = nn.Linear(token_dim, slot_dim)

    def forward(self, slots: torch.Tensor, actions: torch.Tensor):
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
        if self.prev_hidden is None:
            self.prev_hidden = torch.zeros(batch_size * num_slots, self.hidden_dim, device=slots.device)

        # Integrate temporal information of the new set of slots.
        hidden = self.gru(
            slots.reshape(-1, slot_dim),
            self.prev_hidden,
        )
        self.prev_hidden = hidden

        # On the updated hidden states and embedded action, do self-attention to predict the next set of slots.
        action_tokens = self.actions_to_tokens(actions)
        slot_tokens = self.hiddens_to_tokens(hidden).reshape(batch_size, num_slots, self.token_dim)
        tokens = torch.cat((slot_tokens, action_tokens.unsqueeze(1)), dim=1)

        print("tokens.shape:", tokens.shape)

        next_tokens = self.transformer(tokens)

        print("next_tokens.shape:", next_tokens.shape)

        next_slots = self.tokens_to_slots(next_tokens[:, :-1])  # Remove the action token.

        input()

        if self.residual:
            next_slots = next_slots + slots
        return next_slots
