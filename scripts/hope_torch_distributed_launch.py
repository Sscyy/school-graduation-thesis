import json
import os
import sys


def get_launch_cmd():
    afo_spec_str = os.environ.get("AFO_SPEC", "").strip()

    gpu_count      = os.popen("nvidia-smi --list-gpus | wc -l").read().strip()
    nproc_per_node = int(gpu_count) if gpu_count else 1

    if not afo_spec_str or afo_spec_str.lower() in ("null", "none", "{}"):
        # ── 单机模式：AFO_SPEC 未注入，直接用本机所有 GPU ──
        sys.stderr.write(f"[INFO] AFO_SPEC not set, running single-node mode with {nproc_per_node} GPU(s)\n")
        node_rank   = 0
        nnodes      = 1
        master_addr = "127.0.0.1"
        master_port = "29500"

    else:
        # ── 多机模式：从 AFO_SPEC 解析集群信息 ──
        try:
            afo_spec = json.loads(afo_spec_str)

            role = afo_spec["role"]
            if role != "worker":
                sys.stderr.write(f"Error: Role is {role}, expected 'worker'\n")
                sys.exit(1)

            workers   = afo_spec["cluster"]["worker"]
            nnodes    = len(workers)
            # taskId 即当前节点的 rank
            node_rank = int(afo_spec["taskId"])

            # master 取 worker 列表第一个节点的地址和端口
            master_addr, master_port = workers[0].rsplit(":", 1)

        except Exception as e:
            sys.stderr.write(f"Error parsing AFO_SPEC: {str(e)}\n")
            sys.exit(1)

    cmd = (
        "torchrun "
        "--nproc_per_node={} "
        "--nnodes={} "
        "--node_rank={} "
        "--master_addr={} "
        "--master_port={} "
        .format(nproc_per_node, nnodes, node_rank, master_addr, master_port)
    )

    return cmd


if __name__ == "__main__":
    print(get_launch_cmd())
