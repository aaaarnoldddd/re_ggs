import subprocess
import itertools
import re
import sys

def multiple_runs_subprocess():
    kernel_sizes = [3, 5, 7]
    input_sizes = [128, 256]
    dropouts = [0.1, 0.2, 0.3]

    num_experts = [3, 4, 5]
    expert_hidden_units_list = ["[128, 64]", "[256, 128]", "[256, 128, 64]"]
    gate_hidden_units_list = ["[64]", "[64, 128]"]

    tower_hidden_units_list = ["[64]", "[64, 64]", "[128, 64]", "[256, 128, 64]"]
    batch_norms = [True, False]

    learning_rates = [1e-3, 5e-4, 1e-4, 2e-3]
    weight_decays = [1e-5, 1e-4]
    net_dropouts = [0.1, 0.2, 0.3]

    results_file = "results.txt"

    with open(results_file, "w") as f:
        f.write("Experiment Results:\n")
        f.write(
            f"{'kernel_size':<12} | {'input_size':<10} | {'dropout':<8} | {'num_experts':<11} | "
            f"{'expert_units':<14} | {'gate_units':<12} | {'tower_units':<14} | {'batch_norm':<10} | "
            f"{'lr':<8} | {'weight_decay':<12} | {'net_dropout':<10} | {'BestTrainLoss':<14} | {'TestLoss':<10}\n"
        )
        f.write("-" * 190 + "\n")

    for ks, inp, do, ne, eu, gu, tu, bn, lr, wd, nd in itertools.product(
        kernel_sizes, input_sizes, dropouts, num_experts, expert_hidden_units_list, 
        gate_hidden_units_list, tower_hidden_units_list, batch_norms, learning_rates, 
        weight_decays, net_dropouts
    ):
        cmd = (
            f"PYTHONUNBUFFERED=1 python src/train.py "
            f"model.mtl.kernel_size={ks} "
            f"model.mtl.input_size={inp} "
            f"model.mtl.dropout={do} "
            f"model.mtl.num_experts={ne} "
            f"model.mtl.expert_hidden_units='{eu}' "
            f"model.mtl.gate_hidden_units='{gu}' "
            f"model.mtl.tower_hidden_units='{tu}' "
            f"model.mtl.batch_norm={bn} "
            f"model.mtl.learning_rate={lr} "
            f"model.optimizer.weight_decay={wd} "
            f"model.mtl.net_dropout={nd} "
        )

        print(f"Running: {cmd}")

        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # ⚠️ 让 stderr 和 stdout 一起输出，防止进度条错乱
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        best_train_loss = None
        for line in iter(process.stdout.readline, ""):
            sys.stdout.write(line)  # **实时输出**
            sys.stdout.flush()
            match = re.search(r"Best train loss during training:\s*([\d\.]+)", line)
            if match:
                best_train_loss = float(match.group(1))

            match_test = re.search(r"test_loss\s+([\d\.]+)", line)
            if match_test:
                test_loss = float(match_test.group(1))

        process.stdout.close()
        process.wait()

        if best_train_loss is None:
            best_train_loss = "FAILED"

        with open(results_file, "a") as f:
            f.write(f"{ks:<12} | {inp:<10} | {do:<8} | {ne:<11} | {eu:<14} | {gu:<12} | "
                    f"{tu:<14} | {bn:<10} | {lr:<8} | {wd:<12} | {nd:<10} | {best_train_loss:<14} | {test_loss}\n")

        print(f"Results => Best train loss: {best_train_loss}, test_loss: {test_loss}") 

if __name__ == "__main__":
    multiple_runs_subprocess()
