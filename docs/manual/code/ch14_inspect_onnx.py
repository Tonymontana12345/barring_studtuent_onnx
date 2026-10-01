"""14장 완성 코드: ONNX 파일이 어떻게 구성되어 있는지 들여다보고, 공개 전에 개인 정보를 지운다.

    1) 손으로 아주 작은 ONNX(y = x·W + b)를 만들어 구조 익히기
    2) 실제 Student ONNX: 헤더, 입출력, 노드(연산), 가중치(initializer), 외부 데이터 파일
    3) 노드를 PyTorch 모듈(RevIN, 패치 임베딩, 인코더 층, head)별로 묶어 보기
    4) 노드 메타데이터에 남은 소스 경로(stack trace) 찾아서 지우기

실행 (프로젝트 루트에서, 13장 결과 사용):
    python docs/manual/code/ch14_inspect_onnx.py
    python docs/manual/code/ch14_inspect_onnx.py --onnx outputs/manual/onnx/student.onnx --clean outputs/manual/onnx_clean/student.onnx
"""

import argparse
import re
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[3]
DTYPE = {TensorProto.FLOAT: "float32", TensorProto.INT64: "int64", TensorProto.BOOL: "bool"}


# ── 1) 손으로 만드는 가장 작은 ONNX ──────────────────────────────────────────
def build_tiny(path):
    """입력 x[1,2] → MatMul(W[2,2]) → Add(b[2]) → 출력 y[1,2]. 2장의 선형 분류기와 같은 계산."""
    W = numpy_helper.from_array(np.array([[1.0, -1.0], [0.5, 2.0]], dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.array([0.1, -0.1], dtype=np.float32), name="b")
    nodes = [helper.make_node("MatMul", ["x", "W"], ["xw"], name="matmul"),
             helper.make_node("Add", ["xw", "b"], ["y"], name="add")]
    graph = helper.make_graph(
        nodes, "tiny_linear",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        initializer=[W, b])
    # ir_version: 파일 형식 버전. onnx 1.23은 기본값이 14인데 onnxruntime 1.30은 13까지만 읽는다.
    # PyTorch가 만든 파일과 같은 10으로 지정한다. (opset은 연산 버전, IR은 파일 형식 버전으로 서로 다르다)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], producer_name="manual", ir_version=10)
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model


# ── 2) 실제 모델 구성 ────────────────────────────────────────────────────────
def shape_of(value_info):
    return [d.dim_value if d.dim_value else d.dim_param for d in value_info.type.tensor_type.shape.dim]


def numel(tensor):
    return int(np.prod(tensor.dims)) if tensor.dims else 1


def describe(path, top=5):
    model = onnx.load(path, load_external_data=False)      # 구조만 읽는다 (가중치 86 MB는 안 읽음)
    g = model.graph
    print(f"■ 헤더: IR 버전 {model.ir_version}, opset {[(o.domain or 'ai.onnx', o.version) for o in model.opset_import]}, "
          f"만든 도구 {model.producer_name} {model.producer_version}")
    print("■ 입력:", [(i.name, shape_of(i), DTYPE.get(i.type.tensor_type.elem_type)) for i in g.input])
    print("■ 출력:", [(o.name, shape_of(o), DTYPE.get(o.type.tensor_type.elem_type)) for o in g.output])

    ops = Counter(n.op_type for n in g.node)
    print(f"■ 노드(연산) {len(g.node)}개, 종류 {len(ops)}가지:")
    print("   ", ", ".join(f"{k} {v}" for k, v in ops.most_common()))

    total = sum(numel(t) for t in g.initializer)
    external = [t for t in g.initializer if t.data_location == TensorProto.EXTERNAL]
    print(f"■ 가중치(initializer) {len(g.initializer)}개, 숫자 {total:,}개 ({total * 4 / 1e6:.1f} MB)")
    print(f"   그중 외부 파일에 저장 {len(external)}개, .onnx 안에 직접 저장 {len(g.initializer) - len(external)}개(작은 상수)")
    if external:
        info = {e.key: e.value for e in external[0].external_data}
        print(f"   외부 데이터 예: {external[0].name} → 파일 '{info['location']}', 시작 {info['offset']}바이트, 길이 {info['length']}바이트")
    print(f"■ 가장 큰 가중치 {top}개:")
    for t in sorted(g.initializer, key=numel, reverse=True)[:top]:
        print(f"   {t.name:48s} {list(t.dims)}")
    return model


def module_of(node):
    """노드 메타데이터의 name_scopes로 어느 PyTorch 모듈에서 나온 연산인지 알아낸다."""
    scopes = {p.key: p.value for p in node.metadata_props}.get("pkg.torch.onnx.name_scopes", "")
    s = " ".join(re.findall(r"'([^']*)'", scopes))
    if "normalizer" in s:
        return "① normalizer (RevIN)"
    if "patch_embedding" in s or "tokenizer" in s:
        return "②③ 패치 + 임베딩"
    m = re.search(r"block\.(\d+)", s) or re.search(r"block (\d+)", s)
    if m:
        return f"④ encoder 층 {int(m.group(1)):d}"
    if "encoder" in s:
        return "④ encoder (공통: 위치 bias, 마지막 norm)"
    if "head" in s:
        return "⑤ head"
    return "기타 (모양 변환 등)"


def by_module(model):
    groups = OrderedDict()
    for n in model.graph.node:
        groups.setdefault(module_of(n), Counter())[n.op_type] += 1
    for name, ops in groups.items():
        top_ops = ", ".join(f"{k} {v}" for k, v in ops.most_common(6))
        print(f"   {name:34s} 노드 {sum(ops.values()):3d}개  ({top_ops})")


# ── 4) 개인 정보 검사와 정리 ──────────────────────────────────────────────────
def scan_private(model, words=("/Users/", "/home/", "C:\\")):
    hits = Counter()
    for n in model.graph.node:
        for p in n.metadata_props:
            if any(w in p.value for w in words):
                hits[p.key] += 1
    return hits


def clean(src, dst):
    """노드마다 남은 소스 코드 위치(stack trace)를 지운다. 계산에는 영향이 없다."""
    model = onnx.load(src)                              # 가중치까지 읽는다 (다시 저장하려고)
    removed = 0
    for n in model.graph.node:
        keep = [p for p in n.metadata_props if "stack_trace" not in p.key]
        removed += len(n.metadata_props) - len(keep)
        del n.metadata_props[:]
        n.metadata_props.extend(keep)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, dst, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=Path(dst).name + ".data", size_threshold=1024)
    onnx.checker.check_model(dst)
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="outputs/manual/onnx/student.onnx")
    parser.add_argument("--clean", default="outputs/manual/onnx_clean/student.onnx")
    args = parser.parse_args()

    print("[1] 가장 작은 ONNX")
    tiny_path = ROOT / "outputs/manual/onnx/tiny.onnx"
    tiny_path.parent.mkdir(parents=True, exist_ok=True)
    build_tiny(tiny_path)
    import onnxruntime as ort
    y = ort.InferenceSession(str(tiny_path), providers=["CPUExecutionProvider"]).run(
        None, {"x": np.array([[1.0, 2.0]], dtype=np.float32)})[0]
    print(f"   x=[1,2] → y={[round(float(v), 4) for v in y[0]]}  (직접 계산: [1·1+2·0.5+0.1, 1·(-1)+2·2-0.1] = [2.1, 2.9])\n")

    print("[2] Student ONNX 구성")
    model = describe(ROOT / args.onnx)
    print("\n[3] 모듈별 노드")
    by_module(model)

    print("\n[4] 개인 정보 검사")
    hits = scan_private(model)
    print("   메타데이터에서 컴퓨터 경로 발견:", dict(hits) or "없음")
    if hits:
        removed = clean(ROOT / args.onnx, ROOT / args.clean)
        after = scan_private(onnx.load(ROOT / args.clean, load_external_data=False))
        print(f"   stack trace {removed}개 삭제 → {Path(args.clean)} , 남은 경로: {dict(after) or '없음'}")


if __name__ == "__main__":
    main()
