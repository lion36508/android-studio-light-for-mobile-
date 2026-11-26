from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AddSubConfig
from .quantization import initialize_five_symbol_weight_, quantize_weight_ste
from .wavebit_hsra import (
    HolographicRotationMemory,
    resize_context,
    validate_wavebit_options,
    wavebit_depthwise_conv1d,
    wavebit_linear,
    wavebit_quantize,
)


class AddSubLinear(nn.Linear):
    """Training projection with a five-symbol STE weight codebook.

    This module belongs to the training graph. It uses PyTorch autograd and
    floating-point arithmetic during optimization. The deployed inference
    graph must use qmini_addsub.runtime and its integer lookup/add path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        wavebit_options: dict | None = None,
        hadamard: bool = False,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.hadamard = bool(hadamard)
        self.wavebit_options = validate_wavebit_options(wavebit_options)
        initialize_five_symbol_weight_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.wavebit_options is None:
            return F.linear(inputs, quantize_weight_ste(self.weight), self.bias)
        if context is None:
            raise ValueError("WaveBit AddSubLinear requires a block context.")
        options = self.wavebit_options
        return wavebit_linear(
            inputs,
            self.weight,
            self.bias,
            context,
            group_size=options["group_size"],
            ewgs_strength=options["ewgs_strength"],
            octave_temperature=options["octave_temperature"],
            boost_factor=options["octave_boost_factor"],
            hadamard=self.hadamard,
            hadamard_block_size=options["hadamard_block_size"],
            activation_bits=options["activation_bits"],
        )


class CausalConvMixer(nn.Module):
    def __init__(
        self, config: AddSubConfig, wavebit_options: dict | None = None
    ) -> None:
        super().__init__()
        self.kernel_size = config.kernel_size
        self.dilation = 1
        self.wavebit_options = validate_wavebit_options(wavebit_options)
        self.depthwise_weight = nn.Parameter(
            torch.empty(config.d_model, 1, config.kernel_size)
        )
        self.depthwise_bias = nn.Parameter(torch.zeros(config.d_model))
        self.gate_projection = AddSubLinear(
            config.d_model, config.d_model * 2, wavebit_options=wavebit_options
        )
        self.output_projection = AddSubLinear(
            config.d_model, config.d_model, wavebit_options=wavebit_options
        )
        initialize_five_symbol_weight_(self.depthwise_weight)

    def _mix(
        self, padded: torch.Tensor, padded_context: torch.Tensor | None
    ) -> torch.Tensor:
        if self.wavebit_options is None:
            return F.conv1d(
                padded,
                quantize_weight_ste(self.depthwise_weight),
                self.depthwise_bias,
                dilation=self.dilation,
                groups=padded.shape[1],
            )
        if padded_context is None:
            raise ValueError("WaveBit depthwise convolution requires cached context.")
        options = self.wavebit_options
        return wavebit_depthwise_conv1d(
            padded,
            padded_context,
            self.depthwise_weight,
            self.depthwise_bias,
            dilation=self.dilation,
            group_size=options["group_size"],
            ewgs_strength=options["ewgs_strength"],
            octave_temperature=options["octave_temperature"],
            boost_factor=options["octave_boost_factor"],
        )

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        channel_first = inputs.transpose(1, 2)
        padding = self.dilation * (self.kernel_size - 1)
        padded = F.pad(channel_first, (padding, 0))
        padded_context = None
        if self.wavebit_options is not None:
            if context is None or context.shape != inputs.shape:
                raise ValueError("WaveBit convolution context must match its input.")
            padded_context = F.pad(context.transpose(1, 2), (padding, 0))
        mixed = self._mix(padded, padded_context).transpose(1, 2)
        content, gate = self.gate_projection(mixed, context=context).chunk(2, dim=-1)
        activated = F.silu(content) * torch.sigmoid(gate)
        return self.output_projection(activated, context=context)

    def step(
        self,
        inputs: torch.Tensor,
        history: torch.Tensor,
        context: torch.Tensor | None = None,
        context_history: torch.Tensor | None = None,
    ):
        """Process one token and retain only the causal convolution history."""
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("Convolution step expects [batch, 1, features].")
        padding = self.dilation * (self.kernel_size - 1)
        expected_history = (inputs.shape[0], padding, inputs.shape[2])
        if tuple(history.shape) != expected_history:
            raise ValueError(
                f"Convolution cache shape must be {expected_history}, got {tuple(history.shape)}."
            )
        joined = torch.cat((history, inputs), dim=1)
        joined_context = None
        if self.wavebit_options is not None:
            if context is None or context.shape != inputs.shape:
                raise ValueError("WaveBit convolution step requires current-token context.")
            if context_history is None or context_history.shape != history.shape:
                raise ValueError("WaveBit convolution context cache has an invalid shape.")
            joined_context = torch.cat((context_history, context), dim=1)
        mixed = self._mix(
            joined.transpose(1, 2),
            None if joined_context is None else joined_context.transpose(1, 2),
        ).transpose(1, 2)
        content, gate = self.gate_projection(mixed, context=context).chunk(2, dim=-1)
        activated = F.silu(content) * torch.sigmoid(gate)
        updated_history = joined[:, -padding:, :] if padding else joined[:, :0, :]
        output = self.output_projection(activated, context=context)
        if self.wavebit_options is None:
            return output, updated_history
        updated_context_history = (
            joined_context[:, -padding:, :] if padding else joined_context[:, :0, :]
        )
        return output, updated_history, updated_context_history


class RMSNorm(nn.Module):
    """Pre-normalization for stable residual streams."""

    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = inputs * torch.rsqrt(variance + self.eps).to(inputs.dtype)
        return normalized * self.weight.to(inputs.dtype)


class GatedDiagonalStateSpace(nn.Module):
    """Input-dependent diagonal state-space memory with an associative scan."""

    def __init__(
        self, config: AddSubConfig, wavebit_options: dict | None = None
    ) -> None:
        super().__init__()
        self.drive_projection = AddSubLinear(
            config.d_model, config.d_model, wavebit_options=wavebit_options
        )
        self.decay_projection = AddSubLinear(
            config.d_model, config.d_model, wavebit_options=wavebit_options
        )
        self.gate_projection = AddSubLinear(
            config.d_model, config.d_model, wavebit_options=wavebit_options
        )
        self.output_projection = AddSubLinear(
            config.d_model, config.d_model, wavebit_options=wavebit_options
        )
        with torch.no_grad():
            log_times = torch.linspace(
                torch.log(torch.tensor(2.0)),
                torch.log(torch.tensor(float(config.max_context))),
                config.d_model,
            )
            time_constants = torch.exp(log_times)
            rates = time_constants.reciprocal()
            self.decay_projection.weight.zero_()
            self.decay_projection.bias.copy_(torch.log(torch.expm1(rates)))

    @staticmethod
    def parallel_scan(decay: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        """Differentiable O(log(sequence))-depth affine prefix scan."""
        if decay.ndim != 3 or injection.shape != decay.shape:
            raise ValueError("decay and injection must share [batch, sequence, features] shape.")
        if decay.shape[1] < 1:
            raise ValueError("State-space scan requires a non-empty sequence.")
        a = decay.float()
        b = injection.float()
        offset, length = 1, decay.shape[1]
        while offset < length:
            previous_a, previous_b = a, b
            earlier_a = torch.cat(
                (torch.ones_like(previous_a[:, :offset]), previous_a[:, :-offset]),
                dim=1,
            )
            earlier_b = torch.cat(
                (torch.zeros_like(previous_b[:, :offset]), previous_b[:, :-offset]),
                dim=1,
            )
            b = previous_b + previous_a * earlier_b
            a = previous_a * earlier_a
            offset *= 2
        return b

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        drive = torch.tanh(self.drive_projection(inputs, context=context)).float()
        decay = torch.exp(
            -F.softplus(self.decay_projection(inputs, context=context).float())
        ).clamp_min(1e-6)
        states = self.parallel_scan(decay, (1.0 - decay) * drive)
        gated = states.to(dtype=inputs.dtype) * torch.sigmoid(
            self.gate_projection(inputs, context=context)
        )
        return self.output_projection(gated, context=context)

    def step(
        self,
        inputs: torch.Tensor,
        previous_state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one token using the diagonal recurrent state."""
        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("State-space step expects [batch, 1, features].")
        if tuple(previous_state.shape) != (inputs.shape[0], inputs.shape[2]):
            raise ValueError("State-space cache must have shape [batch, features].")
        drive = torch.tanh(
            self.drive_projection(inputs, context=context)
        ).float()[:, 0, :]
        decay = torch.exp(
            -F.softplus(self.decay_projection(inputs, context=context).float())
        ).clamp_min(1e-6)[:, 0, :]
        current_state = decay * previous_state.float() + (1.0 - decay) * drive
        gated = current_state.to(dtype=inputs.dtype).unsqueeze(1)
        gated = gated * torch.sigmoid(self.gate_projection(inputs, context=context))
        return self.output_projection(gated, context=context), current_state


class SwiGLUExpert(nn.Module):
    """Parameter-efficient gated expert MLP used by the sparse router."""

    def __init__(
        self, config: AddSubConfig, wavebit_options: dict | None = None
    ) -> None:
        super().__init__()
        self.wavebit_options = validate_wavebit_options(wavebit_options)
        self.up_gate = AddSubLinear(
            config.d_model, config.d_hidden * 2, wavebit_options=wavebit_options
        )
        self.down = AddSubLinear(
            config.d_hidden, config.d_model, wavebit_options=wavebit_options,
            hadamard=bool(self.wavebit_options and self.wavebit_options["hadamard"]),
        )

    def forward(
        self, inputs: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        content, gate = self.up_gate(inputs, context=context).chunk(2, dim=-1)
        hidden = F.silu(content) * gate
        if self.wavebit_options is None:
            return self.down(hidden)
        if context is None:
            raise ValueError("WaveBit expert requires a block context.")
        hidden_context = resize_context(context, hidden.shape[-1])
        return self.down(hidden, context=hidden_context)


class SparseTop1Experts(nn.Module):
    def __init__(
        self, config: AddSubConfig, wavebit_options: dict | None = None
    ) -> None:
        super().__init__()
        self.n_experts = config.n_experts
        self.router_z_loss_coef = config.router_z_loss_coef
        self.loss_free_bias_update_rate = float(config.moe_bias_update_rate)
        self.loss_free_bias_max_abs = float(config.moe_bias_max_abs)
        self.wavebit_options = validate_wavebit_options(wavebit_options)
        if config.moe_loss_free_bias:
            self.register_buffer(
                "loss_free_router_bias", torch.zeros(config.n_experts)
            )
        else:
            self.register_buffer("loss_free_router_bias", None)
        self.router = AddSubLinear(
            config.d_model, config.n_experts, wavebit_options=wavebit_options
        )
        self.experts = nn.ModuleList(
            [SwiGLUExpert(config, wavebit_options=wavebit_options) for _ in range(config.n_experts)]
        )

    @torch.no_grad()
    def update_loss_free_bias(self, observed_load: torch.Tensor) -> bool:
        """Move non-gradient routing bias toward under-used experts.

        Call once after the optimizer step with this layer's valid-token load
        fractions. The bias affects top-1 selection only; router probabilities
        used to weight expert outputs remain un-biased.
        """
        if self.loss_free_router_bias is None:
            return False
        if not self.training:
            raise RuntimeError("Loss-free router bias may only update in training mode.")
        if not isinstance(observed_load, torch.Tensor):
            raise TypeError("observed_load must be a tensor.")
        if observed_load.shape != (self.n_experts,):
            raise ValueError("observed_load must have one value per expert.")
        load = observed_load.detach().to(
            device=self.loss_free_router_bias.device, dtype=torch.float32
        )
        if not bool(torch.isfinite(load).all()) or bool((load < 0).any()):
            raise ValueError("observed_load must be finite and non-negative.")
        total = load.sum()
        if not bool(torch.isfinite(total)):
            raise ValueError("observed_load total must remain finite.")
        if not bool(total > 0):
            raise ValueError("observed_load must contain at least one routed token.")
        load = load / total
        target = 1.0 / self.n_experts
        direction = torch.where(
            load < target, torch.ones_like(load),
            torch.where(load > target, -torch.ones_like(load), torch.zeros_like(load)),
        )
        self.loss_free_router_bias.add_(direction * self.loss_free_bias_update_rate)
        self.loss_free_router_bias.clamp_(
            min=-self.loss_free_bias_max_abs,
            max=self.loss_free_bias_max_abs,
        )
        return bool(torch.count_nonzero(direction).item())

    def forward(
        self,
        inputs: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(inputs, context=context).float()
        routing_logits = logits
        if self.loss_free_router_bias is not None:
            routing_logits = logits + self.loss_free_router_bias.to(logits.dtype)
        probabilities = torch.softmax(logits, dim=-1)
        selected = torch.argmax(routing_logits, dim=-1)
        selected_probability = probabilities.gather(-1, selected.unsqueeze(-1))

        flat_logits = logits.reshape(-1, self.n_experts)
        flat_probabilities = probabilities.reshape(-1, self.n_experts)
        flat_selected_valid = selected.reshape(-1)
        if token_mask is None:
            valid_logits = flat_logits
            valid_probabilities = flat_probabilities
        else:
            if token_mask.shape != inputs.shape[:2]:
                raise ValueError("MoE token_mask must match [batch, sequence] inputs.")
            valid = token_mask.to(device=inputs.device, dtype=torch.bool).reshape(-1)
            if not bool(valid.any()):
                raise ValueError("MoE token_mask must include at least one valid token.")
            valid_logits = flat_logits[valid]
            valid_probabilities = flat_probabilities[valid]
            flat_selected_valid = flat_selected_valid[valid]

        assignments = F.one_hot(
            flat_selected_valid, num_classes=self.n_experts
        ).to(dtype=torch.float32)
        load = assignments.mean(dim=0)
        importance = valid_probabilities.mean(dim=0)
        balance_loss = self.n_experts * torch.sum(load * importance)

        # The Switch-style z-loss prevents router logits from growing without
        # bound. It is computed only over real (non-padding) tokens.
        log_partition = torch.logsumexp(valid_logits, dim=-1)
        router_z_loss = log_partition.square().mean()

        flat_inputs = inputs.reshape(-1, inputs.shape[-1])
        flat_selected = selected.reshape(-1)
        flat_probability = selected_probability.reshape(-1, 1)
        flat_context = None
        if self.wavebit_options is not None:
            if context is None or context.shape != inputs.shape:
                raise ValueError("WaveBit MoE context must match its input tensor.")
            flat_context = context.reshape(-1, context.shape[-1])
        output = torch.zeros_like(flat_inputs)
        for expert_index, expert in enumerate(self.experts):
            token_indices = torch.nonzero(
                flat_selected == expert_index, as_tuple=False
            ).flatten()
            if token_indices.numel() == 0:
                continue
            expert_inputs = flat_inputs.index_select(0, token_indices)
            expert_probability = flat_probability.index_select(0, token_indices)
            expert_context = (
                None if flat_context is None else flat_context.index_select(0, token_indices)
            )
            expert_output = expert(expert_inputs, context=expert_context) * expert_probability
            output = output.index_copy(0, token_indices, expert_output)
        return (
            output.reshape_as(inputs),
            balance_loss,
            router_z_loss,
            load.detach(),
        )


class AddSubBlock(nn.Module):
    """Pre-norm residual block with optional WaveBit context modulation."""

    def __init__(
        self, config: AddSubConfig, dilation: int = 1,
        wavebit_options: dict | None = None,
    ) -> None:
        super().__init__()
        self.wavebit_options = validate_wavebit_options(wavebit_options)
        self.wavebit_memory = None
        if self.wavebit_options is not None:
            opts = self.wavebit_options
            self.wavebit_memory = HolographicRotationMemory(
                config.d_model,
                rotation=opts["rotation"],
                gamma=opts["gamma"],
                threshold=opts["context_threshold"],
                temperature=opts["context_temperature"],
                state_limit=opts["state_limit"],
            )
        self.conv_norm = RMSNorm(config.d_model)
        self.conv = CausalConvMixer(config, wavebit_options=wavebit_options)
        self.conv.dilation = dilation
        self.state_norm = RMSNorm(config.d_model)
        self.state_space = GatedDiagonalStateSpace(config, wavebit_options=wavebit_options)
        self.expert_norm = RMSNorm(config.d_model)
        self.experts = SparseTop1Experts(config, wavebit_options=wavebit_options)
        self.dropout = nn.Dropout(config.dropout)
        initial_logit = math.log(config.residual_init / (1.0 - config.residual_init))
        self.conv_residual_logit = nn.Parameter(torch.tensor(initial_logit))
        self.state_residual_logit = nn.Parameter(torch.tensor(initial_logit))
        self.expert_residual_logit = nn.Parameter(torch.tensor(initial_logit))

    def forward(
        self, inputs: torch.Tensor, token_mask: torch.Tensor | None = None,
    ):
        context = None
        if self.wavebit_memory is not None:
            context, _ = self.wavebit_memory(inputs)
        conv_update = self.dropout(
            self.conv(self.conv_norm(inputs), context=context)
        )
        outputs = inputs + torch.sigmoid(self.conv_residual_logit) * conv_update
        state_update = self.dropout(
            self.state_space(self.state_norm(outputs), context=context)
        )
        outputs = outputs + torch.sigmoid(self.state_residual_logit) * state_update
        expert_output, balance_loss, router_z_loss, router_load = self.experts(
            self.expert_norm(outputs), token_mask=token_mask, context=context
        )
        outputs = outputs + torch.sigmoid(self.expert_residual_logit) * self.dropout(expert_output)
        if context is None:
            return outputs, balance_loss, router_z_loss, router_load
        return outputs, balance_loss, router_z_loss, router_load, context

    def step(self, inputs: torch.Tensor, cache):
        """One-token update; WaveBit caches retain both rotation and conv context."""
        if self.wavebit_memory is None:
            conv_history, state = cache
            conv_update, conv_history = self.conv.step(
                self.conv_norm(inputs), conv_history
            )
            outputs = inputs + torch.sigmoid(self.conv_residual_logit) * conv_update
            state_update, state = self.state_space.step(
                self.state_norm(outputs), state
            )
            outputs = outputs + torch.sigmoid(self.state_residual_logit) * state_update
            expert_output, _, _, _ = self.experts(self.expert_norm(outputs))
            outputs = outputs + torch.sigmoid(self.expert_residual_logit) * expert_output
            return outputs, (conv_history, state)

        conv_history, conv_context_history, state, wave_state = cache
        context, wave_state = self.wavebit_memory.step(inputs[:, 0, :], wave_state)
        context = context.unsqueeze(1)
        conv_update, conv_history, conv_context_history = self.conv.step(
            self.conv_norm(inputs), conv_history, context, conv_context_history
        )
        outputs = inputs + torch.sigmoid(self.conv_residual_logit) * conv_update
        state_update, state = self.state_space.step(
            self.state_norm(outputs), state, context=context
        )
        outputs = outputs + torch.sigmoid(self.state_residual_logit) * state_update
        expert_output, _, _, _ = self.experts(
            self.expert_norm(outputs), context=context
        )
        outputs = outputs + torch.sigmoid(self.expert_residual_logit) * expert_output
        return outputs, (conv_history, conv_context_history, state, wave_state), context


class AddSubLanguageModel(nn.Module):
    """Independent random-initialized Add/Sub language model for pretraining."""

    def __init__(
        self,
        config: AddSubConfig | None = None,
        seed: int | None = None,
        wavebit_options: dict | None = None,
    ) -> None:
        super().__init__()
        self.config = config or AddSubConfig()
        self.config.validate()
        self.wavebit_options = validate_wavebit_options(wavebit_options)
        model_seed = self.config.seed if seed is None else seed
        if isinstance(model_seed, bool) or not isinstance(model_seed, int) or model_seed < 0:
            raise ValueError("Model seed must be a non-negative integer.")
        # Isolate parameter initialization so constructing a model does not
        # consume or reset the caller's training/dropout RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(model_seed)
            self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        initialize_five_symbol_weight_(self.token_embedding.weight)
        self.blocks = nn.ModuleList(
            [
                AddSubBlock(
                    self.config, dilation=2 ** index,
                    wavebit_options=self.wavebit_options,
                )
                for index in range(self.config.n_layers)
            ]
        )
        self.final_norm = RMSNorm(self.config.d_model)
        self.final_bias = nn.Parameter(torch.zeros(self.config.vocab_size))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[0] < 1 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must have non-empty [batch, sequence] shape.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use an integer torch.int32 or torch.int64 dtype.")
        if input_ids.shape[1] > self.config.max_context:
            raise ValueError("Sequence exceeds configured maximum context.")
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape.")
            if attention_mask.device != input_ids.device:
                raise ValueError("attention_mask and input_ids must be on the same device.")
            attention_mask = attention_mask.to(dtype=torch.bool)
            if not bool(attention_mask.any()):
                raise ValueError("attention_mask must include at least one valid token.")
            if attention_mask.shape[1] > 1 and bool(
                (attention_mask[:, 1:] & ~attention_mask[:, :-1]).any()
            ):
                raise ValueError("attention_mask must be right-padded without internal gaps.")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("Token ID is outside the configured vocabulary.")

        if self.wavebit_options is None:
            embedding_weight = quantize_weight_ste(self.token_embedding.weight)
        else:
            embedding_weight = wavebit_quantize(
                self.token_embedding.weight,
                group_size=self.wavebit_options["group_size"],
                ewgs_strength=self.wavebit_options["ewgs_strength"],
                octave_temperature=self.wavebit_options["octave_temperature"],
                boost_factor=self.wavebit_options["octave_boost_factor"],
            )[0]
        hidden = F.embedding(input_ids, embedding_weight)
        balance_losses = []
        router_z_losses = []
        router_loads = []
        final_context = None
        for block in self.blocks:
            block_result = block(hidden, token_mask=attention_mask)
            if self.wavebit_options is None:
                hidden, balance_loss, router_z_loss, router_load = block_result
            else:
                hidden, balance_loss, router_z_loss, router_load, final_context = block_result
            balance_losses.append(balance_loss)
            router_z_losses.append(router_z_loss)
            router_loads.append(router_load)

        if self.wavebit_options is None:
            tied_output_weight = quantize_weight_ste(self.token_embedding.weight)
            logits = F.linear(self.final_norm(hidden), tied_output_weight, self.final_bias)
        else:
            logits = wavebit_linear(
                self.final_norm(hidden),
                self.token_embedding.weight,
                self.final_bias,
                final_context,
                group_size=self.wavebit_options["group_size"],
                ewgs_strength=self.wavebit_options["ewgs_strength"],
                octave_temperature=self.wavebit_options["octave_temperature"],
                boost_factor=self.wavebit_options["octave_boost_factor"],
            )
        return {
            "logits": logits,
            "aux_loss": torch.stack(balance_losses).mean(),
            "router_z_loss": torch.stack(router_z_losses).mean(),
            # [layer, expert], with each row summing to one. Diagnostic only.
            "router_load": torch.stack(router_loads),
        }

    @torch.no_grad()
    def update_loss_free_router_bias(self, router_load: torch.Tensor) -> bool:
        """Apply one post-optimizer-step bias update to each layer."""
        if router_load.shape != (self.config.n_layers, self.config.n_experts):
            raise ValueError("router_load must have shape [n_layers, n_experts].")
        if not bool(torch.isfinite(router_load).all()) or bool((router_load < 0).any()):
            raise ValueError("router_load must be finite and non-negative.")
        changed = False
        for layer_index, block in enumerate(self.blocks):
            changed = block.experts.update_loss_free_bias(router_load[layer_index]) or changed
        return changed

    def init_cache(
        self,
        batch_size: int = 1,
        device: torch.device | str | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Create independent convolution and recurrent-state caches per block."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        parameter = self.token_embedding.weight
        target_device = parameter.device if device is None else torch.device(device)
        if target_device != parameter.device:
            raise ValueError("Move the model to the target device before initializing its cache.")
        dtype = parameter.dtype
        caches = []
        for block in self.blocks:
            padding = block.conv.dilation * (block.conv.kernel_size - 1)
            history = torch.zeros(
                batch_size, padding, self.config.d_model,
                device=target_device, dtype=dtype,
            )
            state = torch.zeros(
                batch_size, self.config.d_model,
                device=target_device, dtype=torch.float32,
            )
            if self.wavebit_options is None:
                caches.append((history, state))
            else:
                context_history = torch.zeros(
                    batch_size, padding, self.config.d_model,
                    device=target_device, dtype=dtype,
                )
                wave_state = torch.zeros(
                    batch_size, self.config.d_model,
                    device=target_device, dtype=dtype,
                )
                caches.append((history, context_history, state, wave_state))
        return caches

    def forward_step(
        self,
        input_ids: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Evaluate one token per batch item with bounded sequence caches.

        This is a streaming reference path for the PyTorch graph. It does not
        claim the separate integer-only deployed runtime contract.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before using forward_step().")
        if torch.is_grad_enabled():
            raise RuntimeError(
                "forward_step is inference-only; call it under torch.inference_mode()."
            )
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1 or input_ids.shape[0] < 1:
            raise ValueError("forward_step input_ids must have non-empty [batch] tensor shape.")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("forward_step input_ids must use torch.int32 or torch.int64.")
        parameter = self.token_embedding.weight
        batch_size = input_ids.shape[0]
        if input_ids.device != parameter.device:
            raise ValueError("forward_step input_ids must be on the model device.")
        if not isinstance(caches, (list, tuple)) or len(caches) != len(self.blocks):
            raise ValueError("One cache tuple is required per Add/Sub block.")
        for layer_index, (block, cache) in enumerate(zip(self.blocks, caches)):
            padding = block.conv.dilation * (block.conv.kernel_size - 1)
            expected_history = (batch_size, padding, self.config.d_model)
            expected_state = (batch_size, self.config.d_model)
            expected_items = 4 if self.wavebit_options is not None else 2
            if not isinstance(cache, (tuple, list)) or len(cache) != expected_items:
                raise ValueError(
                    f"Cache for block {layer_index} must contain {expected_items} tensors."
                )
            for cache_index, value in enumerate(cache):
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Cache tensor {cache_index} for block {layer_index} is invalid.")
                if value.device != parameter.device:
                    raise ValueError(f"Cache tensor {cache_index} for block {layer_index} is on the wrong device.")
            if tuple(cache[0].shape) != expected_history:
                raise ValueError(f"Convolution cache shape is invalid for block {layer_index}.")
            if tuple(cache[2 if self.wavebit_options is not None else 1].shape) != expected_state:
                raise ValueError(f"State-space cache shape is invalid for block {layer_index}.")
            if cache[0].dtype != parameter.dtype:
                raise TypeError(f"Convolution cache dtype is invalid for block {layer_index}.")
            state_index = 2 if self.wavebit_options is not None else 1
            if cache[state_index].dtype != torch.float32:
                raise TypeError(f"State-space cache dtype must be float32 for block {layer_index}.")
            if self.wavebit_options is not None:
                if tuple(cache[1].shape) != expected_history or cache[1].dtype != parameter.dtype:
                    raise ValueError(f"WaveBit context cache is invalid for block {layer_index}.")
                if tuple(cache[3].shape) != expected_state or cache[3].dtype != parameter.dtype:
                    raise ValueError(f"WaveBit recurrent cache is invalid for block {layer_index}.")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError("Token ID is outside the configured vocabulary.")

        if self.wavebit_options is None:
            embedding_weight = quantize_weight_ste(self.token_embedding.weight)
        else:
            embedding_weight = wavebit_quantize(
                self.token_embedding.weight,
                group_size=self.wavebit_options["group_size"],
                ewgs_strength=self.wavebit_options["ewgs_strength"],
                octave_temperature=self.wavebit_options["octave_temperature"],
                boost_factor=self.wavebit_options["octave_boost_factor"],
            )[0]
        hidden = F.embedding(input_ids, embedding_weight).unsqueeze(1)
        updated_caches = []
        final_context = None
        for block, cache in zip(self.blocks, caches):
            if self.wavebit_options is None:
                hidden, updated = block.step(hidden, cache)
            else:
                hidden, updated, final_context = block.step(hidden, cache)
            updated_caches.append(updated)
        hidden = self.final_norm(hidden[:, 0, :])
        if self.wavebit_options is None:
            tied_output_weight = quantize_weight_ste(self.token_embedding.weight)
            logits = F.linear(hidden, tied_output_weight, self.final_bias)
        else:
            logits = wavebit_linear(
                hidden,
                self.token_embedding.weight,
                self.final_bias,
                final_context[:, 0, :],
                group_size=self.wavebit_options["group_size"],
                ewgs_strength=self.wavebit_options["ewgs_strength"],
                octave_temperature=self.wavebit_options["octave_temperature"],
                boost_factor=self.wavebit_options["octave_boost_factor"],
            )
        return logits, updated_caches

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
