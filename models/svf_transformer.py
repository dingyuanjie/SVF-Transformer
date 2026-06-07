from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class SVFTransformerConfig:
    vocab_size: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 512
    core_slots: int = 4
    memory_size: int = 256
    attractor_strength: float = 0.05
    drift_scale: float = 0.1
    conservation_weight: float = 0.01
    drift_weight: float = 0.001
    use_memory: bool = True
    use_persistent_core: bool = True
    use_structural_dynamics: bool = True
    use_structural_loss: bool = True
    use_slot_routing: bool = False
    use_top1_routing: bool = False
    use_slot_diversity_loss: bool = False
    slot_diversity_weight: float = 0.01
    use_slot_balance_loss: bool = False
    slot_balance_weight: float = 0.01


@dataclass
class SVFTransformerOutput:
    logits: Tensor
    loss: Optional[Tensor]
    ce_loss: Optional[Tensor]
    core_state: Tensor
    structural_energy: Tensor
    conservation_loss: Tensor
    drift_loss: Tensor
    slot_diversity_loss: Tensor
    slot_balance_loss: Tensor
    slot_routing_weights: Optional[Tensor]
    slot_read_weights: Optional[Tensor]


class RingBufferMemory(nn.Module):
    """Fixed-size memory bank with attention read and detached writes."""

    def __init__(self, memory_size: int, d_model: int) -> None:
        super().__init__()
        self.memory_size = memory_size
        self.d_model = d_model
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Linear(d_model * 2, d_model)
        self.register_buffer("memory", torch.zeros(memory_size, d_model))
        self.register_buffer("write_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("filled", torch.zeros((), dtype=torch.long))

    def reset(self) -> None:
        self.memory.zero_()
        self.write_index.zero_()
        self.filled.zero_()

    @property
    def available(self) -> int:
        return int(self.filled.item())

    def read(self, hidden: Tensor) -> Tensor:
        if self.available == 0:
            return torch.zeros_like(hidden)

        memory = self.memory[: self.available].detach().clone().to(hidden.device)
        q = self.query(hidden)
        k = self.key(memory)
        v = self.value(memory)
        scores = torch.matmul(q, k.transpose(0, 1)) / (self.d_model**0.5)
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, v)
        gate = torch.sigmoid(self.gate(torch.cat([hidden, context], dim=-1)))
        return gate * context

    @torch.no_grad()
    def write(self, hidden: Tensor) -> None:
        summary = hidden.detach().mean(dim=(0, 1)).to(self.memory.device)
        idx = int(self.write_index.item())
        self.memory[idx].copy_(summary)
        self.write_index.fill_((idx + 1) % self.memory_size)
        self.filled.fill_(min(self.available + 1, self.memory_size))


class PersistentCore(nn.Module):
    """Stateful structural core with bounded drift and attractor pull."""

    def __init__(self, config: SVFTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.initial_core = nn.Parameter(torch.randn(config.core_slots, config.d_model) * 0.02)
        self.attractor = nn.Parameter(torch.zeros(config.core_slots, config.d_model))
        self.summary_proj = nn.Linear(config.d_model, config.d_model)
        self.slot_router = nn.Linear(config.d_model, config.d_model, bias=False)
        self.slot_keys = nn.Parameter(torch.randn(config.core_slots, config.d_model) * 0.02)
        self.slot_embeddings = nn.Parameter(torch.randn(config.core_slots, config.d_model) * 0.02)
        self.slot_read_query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.core_update = nn.GRUCell(config.d_model, config.d_model)
        self.drift = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.hidden_gate = nn.Linear(config.d_model * 2, config.d_model)
        self.core_to_hidden = nn.Linear(config.d_model, config.d_model)

    def fresh_state(self, batch_size: int, device: torch.device) -> Tensor:
        return self.initial_core.unsqueeze(0).expand(batch_size, -1, -1).to(device)

    def forward(
        self,
        hidden: Tensor,
        core_state: Optional[Tensor],
        use_structural_dynamics: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        batch_size = hidden.size(0)
        if core_state is None:
            core_state = self.fresh_state(batch_size, hidden.device)

        summary = self.summary_proj(hidden.mean(dim=1))
        flat_core = core_state.reshape(batch_size * self.config.core_slots, self.config.d_model)
        repeated_summary = summary[:, None, :].expand(-1, self.config.core_slots, -1)
        slot_routing_weights: Optional[Tensor] = None
        slot_read_weights: Optional[Tensor] = None
        if self.config.use_slot_routing:
            route_query = self.slot_router(summary)
            route_logits = torch.einsum("bd,sd->bs", route_query, self.slot_keys) / (self.config.d_model**0.5)
            route_probabilities = torch.softmax(route_logits, dim=-1)
            if self.config.use_top1_routing:
                top_indices = route_probabilities.argmax(dim=-1, keepdim=True)
                hard_assignments = torch.zeros_like(route_probabilities).scatter_(1, top_indices, 1.0)
                # Straight-through routing keeps a hard write decision with soft gradients.
                slot_routing_weights = hard_assignments - route_probabilities.detach() + route_probabilities
            else:
                slot_routing_weights = route_probabilities
            repeated_summary = repeated_summary + self.slot_embeddings.unsqueeze(0)
            repeated_summary = repeated_summary * slot_routing_weights.unsqueeze(-1)
        repeated_summary = repeated_summary.reshape_as(flat_core)

        updated = self.core_update(repeated_summary, flat_core)
        updated = updated.view(batch_size, self.config.core_slots, self.config.d_model)

        if use_structural_dynamics:
            drift = torch.tanh(self.drift(updated)) * self.config.drift_scale
            attractor = self.attractor.unsqueeze(0)
            next_core = updated + drift - self.config.attractor_strength * (updated - attractor)
        else:
            drift = torch.zeros_like(updated)
            next_core = updated

        old_energy = core_state.pow(2).mean(dim=(1, 2))
        new_energy = next_core.pow(2).mean(dim=(1, 2))
        conservation_loss = (new_energy - old_energy).pow(2).mean()
        drift_loss = drift.pow(2).mean()
        slot_diversity_loss = hidden.new_zeros(())
        slot_balance_loss = hidden.new_zeros(())
        if self.config.use_slot_diversity_loss and self.config.core_slots > 1:
            normalized_core = F.normalize(next_core, dim=-1)
            similarity = torch.matmul(normalized_core, normalized_core.transpose(1, 2))
            eye = torch.eye(self.config.core_slots, device=hidden.device, dtype=torch.bool).unsqueeze(0)
            offdiag = similarity.masked_select(~eye)
            slot_diversity_loss = offdiag.pow(2).mean()
        if self.config.use_slot_balance_loss and slot_routing_weights is not None and self.config.core_slots > 1:
            target = torch.full_like(slot_routing_weights.mean(dim=0), 1.0 / self.config.core_slots)
            slot_balance_loss = (slot_routing_weights.mean(dim=0) - target).pow(2).mean()

        if slot_routing_weights is not None:
            read_query = self.slot_read_query(summary)
            read_logits = torch.einsum("bd,bsd->bs", read_query, next_core) / (self.config.d_model**0.5)
            slot_read_weights = torch.softmax(read_logits, dim=-1)
            core_summary = (slot_read_weights.unsqueeze(-1) * next_core).sum(dim=1)
        else:
            core_summary = next_core.mean(dim=1)

        core_context = self.core_to_hidden(core_summary).unsqueeze(1)
        gate = torch.sigmoid(self.hidden_gate(torch.cat([hidden, core_context.expand_as(hidden)], dim=-1)))
        hidden = hidden + gate * core_context
        return (
            hidden,
            next_core,
            new_energy.mean(),
            conservation_loss,
            drift_loss,
            slot_diversity_loss,
            slot_balance_loss,
            slot_routing_weights,
            slot_read_weights,
        )


class IdentityCore(nn.Module):
    """Fallback core used by the baseline and memory-only ablations."""

    def __init__(self, config: SVFTransformerConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        hidden: Tensor,
        core_state: Optional[Tensor],
        use_structural_dynamics: bool = True,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        batch_size = hidden.size(0)
        device = hidden.device
        next_core = torch.zeros(
            batch_size,
            self.config.core_slots,
            self.config.d_model,
            device=device,
            dtype=hidden.dtype,
        )
        zero = hidden.new_zeros(())
        return hidden, next_core, zero, zero, zero, zero, zero, None, None


def build_config_for_variant(
    base_config: SVFTransformerConfig,
    variant: str,
) -> SVFTransformerConfig:
    variant_key = variant.strip().lower().replace("-", "_")
    if variant_key == "baseline":
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": False,
                "use_persistent_core": False,
                "use_structural_dynamics": False,
                "use_structural_loss": False,
            }
        )
    if variant_key == "memory":
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": True,
                "use_persistent_core": False,
                "use_structural_dynamics": False,
                "use_structural_loss": False,
            }
        )
    if variant_key in {"persistent_core", "core"}:
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": False,
                "use_persistent_core": True,
                "use_structural_dynamics": False,
                "use_structural_loss": False,
            }
        )
    if variant_key in {"core_dynamics", "persistent_core_dynamics"}:
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": False,
                "use_persistent_core": True,
                "use_structural_dynamics": True,
                "use_structural_loss": False,
            }
        )
    if variant_key in {"specialized_core", "slot_routed_core"}:
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": False,
                "use_persistent_core": True,
                "use_structural_dynamics": False,
                "use_structural_loss": False,
                "use_slot_routing": True,
                "use_slot_diversity_loss": True,
            }
        )
    if variant_key in {"specialized_core_dynamics", "slot_routed_core_dynamics"}:
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": False,
                "use_persistent_core": True,
                "use_structural_dynamics": True,
                "use_structural_loss": False,
                "use_slot_routing": True,
                "use_slot_diversity_loss": True,
            }
        )
    if variant_key in {"memory_core", "core_memory"}:
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": True,
                "use_persistent_core": True,
                "use_structural_dynamics": False,
                "use_structural_loss": False,
            }
        )
    if variant_key in {"svf", "full_svf"}:
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": True,
                "use_persistent_core": True,
                "use_structural_dynamics": True,
                "use_structural_loss": True,
            }
        )
    if variant_key in {"specialized_svf", "slot_routed_svf"}:
        return SVFTransformerConfig(
            **{
                **base_config.__dict__,
                "use_memory": True,
                "use_persistent_core": True,
                "use_structural_dynamics": True,
                "use_structural_loss": True,
                "use_slot_routing": True,
                "use_slot_diversity_loss": True,
            }
        )
    raise ValueError(
        "Unknown variant. Expected one of: "
        "baseline, memory, persistent_core, core_dynamics, specialized_core, "
        "specialized_core_dynamics, memory_core, svf, specialized_svf."
    )


class SVFTransformer(nn.Module):
    def __init__(self, config: SVFTransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.memory = RingBufferMemory(config.memory_size, config.d_model)
        self.core = PersistentCore(config) if config.use_persistent_core else IdentityCore(config)
        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def reset_state(self) -> None:
        self.memory.reset()

    def forward(
        self,
        input_ids: Tensor,
        targets: Optional[Tensor] = None,
        core_state: Optional[Tensor] = None,
        use_memory: bool = True,
        write_memory: bool = False,
    ) -> SVFTransformerOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")

        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.backbone(hidden, mask=causal_mask)

        if use_memory:
            hidden = hidden + self.memory.read(hidden)

        (
            hidden,
            next_core,
            energy,
            conservation_loss,
            drift_loss,
            slot_diversity_loss,
            slot_balance_loss,
            slot_routing_weights,
            slot_read_weights,
        ) = self.core(
            hidden,
            core_state,
            use_structural_dynamics=self.config.use_structural_dynamics,
        )
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)

        loss = None
        ce_loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss = ce_loss
            if self.config.use_structural_loss:
                loss = (
                    loss
                    + self.config.conservation_weight * conservation_loss
                    + self.config.drift_weight * drift_loss
                )
            if self.config.use_slot_diversity_loss:
                loss = loss + self.config.slot_diversity_weight * slot_diversity_loss
            if self.config.use_slot_balance_loss:
                loss = loss + self.config.slot_balance_weight * slot_balance_loss

        if write_memory:
            self.memory.write(hidden)

        return SVFTransformerOutput(
            logits=logits,
            loss=loss,
            ce_loss=ce_loss,
            core_state=next_core,
            structural_energy=energy,
            conservation_loss=conservation_loss,
            drift_loss=drift_loss,
            slot_diversity_loss=slot_diversity_loss,
            slot_balance_loss=slot_balance_loss,
            slot_routing_weights=slot_routing_weights,
            slot_read_weights=slot_read_weights,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        self.eval()
        core_state = None
        tokens = input_ids
        for _ in range(max_new_tokens):
            window = tokens[:, -self.config.max_seq_len :]
            out = self(window, core_state=core_state, use_memory=True, write_memory=False)
            core_state = out.core_state
            logits = out.logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
        return tokens
