============================================================
Stage I SFT Overfitting Test
============================================================
  Samples:  5
  Steps:    500
  Data:     /root/autodl-tmp/data/full5000/sft_tiny_5.jsonl

Device: cuda

[1/5] Loading Gemma3-ISAC model...
`torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1065/1065 [00:00<00:00, 4079.61it/s]
/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:1348: UserWarning: Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, but `ensure_weight_tying` is not set to True. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. Check the discussion here: https://github.com/huggingface/peft/issues/2777
  warnings.warn(msg)
  Model loaded in 34.2s
  [DIAG] Parameter devices: {'cuda:0'}
  Trainable params: 2,045,428,704

  [DIAG] First 5 trainable parameters:
    1. base_model.base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight  |  device=cuda:0  |  shape=(16, 1152)  |  dtype=torch.float32
    2. base_model.base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.lora_B.default.weight  |  device=cuda:0  |  shape=(1152, 16)  |  dtype=torch.float32
    3. base_model.base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.lora_A.default.weight  |  device=cuda:0  |  shape=(16, 1152)  |  dtype=torch.float32
    4. base_model.base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.lora_B.default.weight  |  device=cuda:0  |  shape=(1152, 16)  |  dtype=torch.float32
    5. base_model.base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight  |  device=cuda:0  |  shape=(16, 1152)  |  dtype=torch.float32

  [DIAG] Projection head params in model.named_parameters(): 16
    projection_head.readout.readout.0.weight  |  device=cuda:0  |  requires_grad=True
    projection_head.readout.readout.0.bias  |  device=cuda:0  |  requires_grad=True
    projection_head.readout.readout.2.weight  |  device=cuda:0  |  requires_grad=True

  [DIAG] Trainable params in model.base_model.named_parameters(): 548
    base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight  |  device=cuda:0  |  shape=(16, 1152)
    base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.lora_B.default.weight  |  device=cuda:0  |  shape=(1152, 16)
    base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.lora_A.default.weight  |  device=cuda:0  |  shape=(16, 1152)
    base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.lora_B.default.weight  |  device=cuda:0  |  shape=(1152, 16)
    base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight  |  device=cuda:0  |  shape=(16, 1152)

[2/5] Loading dataset...
  Dataset size: 5

[3/5] Setting up optimizer...
  Optimizing: Projection Head (16 tensors), LoRA (548 tensors)
  [DIAG] Optimizer has 564 param groups with 2,045,428,704 total params

[4/5] Running overfitting loop (500 steps)...
Overfitting:   0%|                                                                                                                                                               | 0/500 [00:00<?, ?it/s]
  [DIAG] === Step 0 forward diagnostics ===
  hidden_states norm: 6176.0000 (expect > 100)
  control_states norm: 292.0000 (expect > 10)
  delta_q norm: 2.4651  delta_a norm: 2.8214  delta_p norm: 0.7373
  control_mask True count: 8 (expect 8)

  [DIAG] === Step 0 gradient diagnostics ===
  Top 5 gradients by norm:
    base_model.base_model.model.model.language_model.embed_tokens.modules_to_save.default.weight: grad_norm=924.000000
    projection_head.readout.readout.0.weight: grad_norm=174.884293
    projection_head.readout.readout.3.weight: grad_norm=105.053017
    projection_head.mlp.net.6.weight: grad_norm=38.332428
    base_model.base_model.model.lm_head.modules_to_save.default.weight: grad_norm=27.875000
  Parameters with ZERO gradient:
    base_model.base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight
    base_model.base_model.model.model.language_model.layers.0.self_attn.k_proj.lora_A.default.weight
    base_model.base_model.model.model.language_model.layers.0.self_attn.v_proj.lora_A.default.weight
    base_model.base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_A.default.weight
    base_model.base_model.model.model.language_model.layers.1.self_attn.q_proj.lora_A.default.weight
    ... and 195 more

  [DIAG] === Step 0 weight change diagnostics ===
    base_model.base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.lora_A.default.weight: NO CHANGE (Δ=0)
    base_model.base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.k_proj.lora_B.default.weight: NO CHANGE (Δ=0)
    base_model.base_model.model.model.vision_tower.vision_model.encoder.layers.0.self_attn.v_proj.lora_A.default.weight: NO CHANGE (Δ=0)
    base_model.base_model.model.model.language_model.embed_tokens.modules_to_save.default.weight: max |Δw| = 0.000244
    base_model.base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight: max |Δw| = 0.000200
    base_model.base_model.model.model.language_model.layers.0.self_attn.k_proj.lora_B.default.weight: max |Δw| = 0.000200
    base_model.base_model.model.model.language_model.layers.0.self_attn.v_proj.lora_B.default.weight: max |Δw| = 0.000200
    base_model.base_model.model.model.language_model.layers.0.self_attn.o_proj.lora_B.default.weight: max |Δw| = 0.000200
  Result: 202 params changed, 362 UNCHANGED
  ✗ Some parameters did NOT update after optimizer.step()
Overfitting:   1%|█                                                                                                              | 5/500 [00:09<14:42,  1.78s/it, total=27.0448, sft=1.3750, ctl=51.3396]
  [DIAG] === Step 5 lora_A gradient check ===
  lora_A with gradient: 192
  lora_A still zero:    0
  ✓ lora_A gradients emerging (B no longer zero → A now receives gradients)
Overfitting:   5%|                   | 26/500 [00:44<13:11,  1.67s/it, total=7.0136, sft=0.3730, ctl=13.2811]
54/500 [01:31<12:24,  1.67s/it, total=24.4441, sft=0.1104, ctl=48.6676]
58/500 [01:38<12:18,  1.67s/it, total=7.1671, sft=0.2236, ctl=13.8869

107/500 [02:59<10:56,  1.67s/it, total=0.3619, sft=0.0173, ctl=0.6892]