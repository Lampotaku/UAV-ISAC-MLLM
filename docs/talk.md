Loading weights: 100%|| 1065/1065 [00:00<00:00, 5497.26it/s]
/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/peft/tuners/tuners_utils.py:1348: UserWarning: Model has `tie_word_embeddings=True` and a tied layer is part of the adapter, but `ensure_weight_tying` is not set to True. This can lead to complications, for example when merging the adapter or converting your model to formats other than safetensors. Check the discussion here: https://github.com/huggingface/peft/issues/2777
  warnings.warn(msg)
Traceback (most recent call last):
  File "/root/UAV-ISAC-MLLM/src/training/train_sft.py", line 633, in <module>
    train_stage1(args.config, args.data_dir, resume_from=args.resume_from)
  File "/root/UAV-ISAC-MLLM/src/training/train_sft.py", line 190, in train_stage1
    model = Gemma3ISAC.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/UAV-ISAC-MLLM/src/model/gemma_isac.py", line 514, in from_pretrained
    causal_lm.lm_head.weight = embed.weight
    ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1993, in __setattr__
    self.register_parameter(name, value)
  File "/root/miniconda3/envs/uavmllm/lib/python3.11/site-packages/torch/nn/modules/module.py", line 620, in register_parameter
    raise KeyError(f"attribute '{name}' already exists")
KeyError: "attribute 'weight' already exists"