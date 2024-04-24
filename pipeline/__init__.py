import os
import subprocess
from typing import List


class RunPipeline:
    def __init__(self, program_file: str):
        self.python = "python"
        self.program_file = program_file

    def run(self, args: List):
        subprocess.run([self.python, self.program_file, *args], shell=False)

    def run_list(self, args_list: List, gpu_index=0, num_gpus=1):
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_index}"
        for i in range(0, len(args_list), num_gpus):
            self.run(args_list[i + gpu_index])
