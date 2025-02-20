import subprocess
import os

input_fasta = "msa/1914.fasta"  # sequence
output_a3m = "output.a3m"    # msa
database_path = "/Users/hahayes/Desktop/Capstone/hh-suite/databases/uniclust30_2016_09/uniclust30_2016_09" # database

# HHblits command
hhblits_command = [
    "hhblits", 
    "cpu", "4",
    "-i", input_fasta,          # sequence
    "-d", database_path,        # database
    "-oa3m", output_a3m,        # msa
    "-norealign",               # norealign
    "-n", "3"                   # iteration
]

# run command
try:
    subprocess.run(hhblits_command, check=True)
    print(f"MSA 文件已生成: {output_a3m}")
except subprocess.CalledProcessError as e:
    print(f"运行 HHblits 时出错: {e}")