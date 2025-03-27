import os
from pathlib import Path

resType = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V","X","-"]
def process_files():
    msaPath = os.path.join(Path(__file__).parent, "msa_raw_inference", "MSA.a3m")
    print(msaPath)
    try:
        temp_output = []
        with open(msaPath, 'r') as f: 
            for i, line in enumerate(f):
                if i % 2 == 1:
                    line = line.replace('B', '-').replace('Z', '-').replace('U', '-').replace('O', '-')
                temp_output.append(str(line))

        with open(msaPath, 'w') as f:
            f.writelines(temp_output)
            print("111111")


    except Exception as e:
        print(e)

# if __name__ == "__main__":
#     input_dirs = ['model/input_seqs', 'model/msa_raw']
    
#     for dir_path in input_dirs:
#         print(f"\nProcessing files in {dir_path}")
#         process_files(dir_path)
