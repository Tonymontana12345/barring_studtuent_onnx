"""Model construction and strict checkpoint loading."""

import torch
from momentfm import MOMENTPipeline
from peft import LoraConfig, inject_adapter_in_model

from .data import make_moment_chunks

LORA_CONFIG = {"r": 8, "lora_alpha": 16, "lora_dropout": .05,
               "target_modules": ["q", "v"], "bias": "none", "task_type": "FEATURE_EXTRACTION"}


def create_model(model_dir, teacher=False, lora_config=None):
    # [버그 수정 기록: 기존 Teacher의 LoRA가 학습되지 않던 문제]
    #
    # 증상
    #   기존 teacher_lora_best.pt의 LoRA B 행렬 24개가 전부 0(초기값 그대로)이었다.
    #   LoRA 파라미터 48개(A/B) 모두 학습 중 grad=None이었고, 실제로는 분류 헤드만 학습됐다.
    #   에러가 나지 않아서 20 epoch 동안 아무도 몰랐다.
    #
    # 원인
    #   1) 이전 코드(train_teacher_lora.py)는 model_kwargs에 enable_gradient_checkpointing을
    #      지정하지 않았다. MOMENT는 기본값 True일 때 encoder.gradient_checkpointing_enable()을
    #      인자 없이 호출하고, 이 경우 reentrant 방식 checkpoint가 사용된다.
    #   2) reentrant checkpoint는 "블록 입력 텐서 중 하나라도 requires_grad=True"일 때만
    #      역전파 그래프를 만든다. 블록 안의 파라미터(LoRA)는 입력으로 치지 않는다.
    #   3) embedder(patch embedding)를 고정했기 때문에 인코더 입력은 requires_grad=False였다.
    #      → 인코더 출력이 그래프에서 끊겨 역전파가 헤드에서 멈췄고 LoRA에는 기울기가 오지 않았다.
    #   (부수 효과: 역전파가 인코더를 통과하지 않아 학습이 빨랐다. 수정 후 학습이 느려진 이유다.)
    #
    # 수정
    #   - MOMENT 자체 checkpointing은 끄고(enable_gradient_checkpointing=False),
    #     Teacher에서만 use_reentrant=False로 다시 켠다(아래 teacher 분기).
    #     non-reentrant 방식은 입력이 아니라 파라미터 기준으로 그래프를 유지해 LoRA 기울기가 전달된다.
    #   - Student는 backbone을 고정하고 헤드만 학습하므로 checkpointing이 필요 없어 끈 상태로 둔다.
    #   - 재발 방지: training.validate_gradients()가 매 step 학습 대상 파라미터의 기울기
    #     존재/유한성을 검사해, 같은 문제가 생기면 조용히 넘어가지 않고 즉시 실패한다.
    model = MOMENTPipeline.from_pretrained(
        str(model_dir), local_files_only=True,
        model_kwargs={"task_name": "classification", "n_channels": 1, "num_class": 2,
                      "freeze_encoder": True, "freeze_embedder": True,
                      "enable_gradient_checkpointing": False},
    )
    model.init()
    if teacher:
        config = LORA_CONFIG if lora_config is None else lora_config
        model = inject_adapter_in_model(LoraConfig(**config), model)
        # Reentrant checkpointing drops LoRA gradients when the frozen embedder's
        # output has requires_grad=False. Non-reentrant checkpointing preserves them.
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("head.") or (teacher and "lora_" in name)
    return model


def load_model(model_dir, checkpoint_path, device="cpu", teacher=False):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("lora_config") if teacher else None
    if teacher and config is None:
        config = dict(LORA_CONFIG)
        # Original checkpoints store the LoRA scalars separately.
        for old, new in (("lora_r", "r"), ("lora_alpha", "lora_alpha"), ("lora_dropout", "lora_dropout")):
            if old in checkpoint:
                config[new] = checkpoint[old]
    model = create_model(model_dir, teacher=teacher, lora_config=config)
    state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def window_logits(model, batch_x):
    chunks, mask, _ = make_moment_chunks(batch_x)
    logits = model(x_enc=chunks, input_mask=mask).logits
    return logits.reshape(batch_x.shape[0], -1, 2).mean(dim=1)
