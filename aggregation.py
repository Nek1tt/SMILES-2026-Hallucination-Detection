from __future__ import annotations
import torch

N_LAYERS = 4  # last 4 transformer layers, last token each -> 4*896 = 3584-dim

def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    real_positions = attention_mask.nonzero(as_tuple=False)
    last_pos       = int(real_positions[-1].item())
    transformer_layers = hidden_states[1:]           # (24, seq_len, 896)
    selected           = transformer_layers[-N_LAYERS:]  # (4, seq_len, 896)
    last_tokens = [selected[i][last_pos] for i in range(N_LAYERS)]
    return torch.cat(last_tokens, dim=0)             # (3584,)

def extract_geometric_features(hidden_states, attention_mask):
    return torch.zeros(0, device=hidden_states.device)

def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    attention_mask = attention_mask.to(hidden_states.device)
    return aggregate(hidden_states, attention_mask)
