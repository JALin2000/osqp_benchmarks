import subprocess
import sys
import time

args_list = [
    ["--loss", "log_convergence", "--ckpt", "learned_osqp/checkpoints/best_model_log_convergence_mul.pt"],
    # ["--loss", "spectral_radius", "--ckpt", "learned_osqp/checkpoints/best_model_spectral_radius.pt"],
]

for args in args_list:
    subprocess.run(
        [sys.executable, "learned_osqp/train.py", *args],
        check=True
    )
    time.sleep(20)