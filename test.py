import torch

ckpt = torch.load("/Users/ijunhan/vib-onnx/outputs/checkpoints/student_kd_best.pt", map_location="cpu", weights_only=True)
print(type(ckpt))

if isinstance(ckpt, dict):
    print("키 개수:", len(ckpt))
    for k, v in list(ckpt.items())[:30]:
        print(k, tuple(v.shape) if hasattr(v, "shape") else type(v))