from modeling.autoencoder.base import Autoencoder
from modeling.autoencoder.cnn import CnnEncoder, CnnDecoder
import torch
from typing import Dict, Optional, Tuple


def get_embedding_dim(model, image_size: Tuple[int, int]) -> int:
    x = torch.zeros(1, 3, *image_size)
    with torch.no_grad():
        out = model(x).flatten(start_dim=1)
    return out.shape[-1]


class Cnn(Autoencoder):
    def __init__(self, encoder: CnnEncoder, decoder: CnnDecoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = get_embedding_dim(encoder, image_size=decoder.keywords["image_size"])
        self.decoder = decoder(embedding_dim=self.embedding_dim)

    @property
    def num_slots(self) -> int:
        return 1

    @property
    def slot_dim(self) -> int:
        return self.embedding_dim

    def encode(self, images: torch.Tensor, actions: torch.Tensor, prior_slots: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, sequence_length, _, _, _ = images.size()
        images = images.flatten(0, 1)
        embeddings = self.encoder(images).flatten(start_dim=1)
        slots_sequence = embeddings.view(batch_size, sequence_length, self.num_slots, -1)
        return slots_sequence

    def decode(self, slots: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, sequence_length, num_slots, slot_dim = slots.size()
        reconstructions = self.decoder(slots.flatten(end_dim=1)).view(batch_size, sequence_length, 3, *self.decoder.image_size)
        return {"reconstructions": reconstructions.clamp(0, 1)}

