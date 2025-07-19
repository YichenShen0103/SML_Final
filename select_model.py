import subprocess
import re
import shutil
import os

# 训练-评测执行次数
N = 1200

# 模型默认保存路径（train.py保存的路径）
MODEL_PATH = ".cache/best_model.pth"

# 用于保存最好模型的路径
BEST_MODEL_PATH = "checkpoint/final_LTS.pth"

best_acc = 0.0
best_run = -1


def run_train():
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True


def run_infer():
    result = subprocess.run(["python", "infer.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr)
        return None
    # 从输出中提取准确率 Accuracy: xx.xx%
    output = result.stdout.strip()
    match = re.search(r"Accuracy:\s*([0-9.]+)%", output)
    if match:
        acc = float(match.group(1))
        return acc
    else:
        return None


def main():
    global best_acc, best_run

    for i in range(N):
        print(f"=== 第 {i+1} 次训练评测 ===")
        success = run_train()
        if not success:
            print("跳过此次训练")
            continue
        acc = run_infer()
        if acc is None:
            print("跳过此次评测")
            continue

        if acc > best_acc:
            print(f"当前准确率 {acc}% 高于历史最高 {best_acc}%，保存模型")
            best_acc = acc
            best_run = i + 1
            if os.path.exists(MODEL_PATH):
                shutil.copyfile(MODEL_PATH, BEST_MODEL_PATH)
            else:
                print("警告：模型文件未找到，无法保存")
        else:
            print(f"当前准确率 {acc}% 未超过历史最高 {best_acc}%，不保存模型")

    print(f"训练结束，最高准确率：{best_acc}%（第{best_run}次）")
    print(f"最好模型已保存为：{BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
