import os
resType = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V","X","-"]
def process_files(input_dir, output_dir=None):
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.a3m') or filename.endswith('.fasta'): 
            input_path = os.path.join(input_dir, filename)
            
            with open(input_path, 'r') as f:
                lines = f.readlines()
            
            for i in range(len(lines)):
                if i % 2 == 1: 
                    lines[i] = lines[i].replace('B', '-')
                    lines[i] = lines[i].replace('Z', '-')
                    lines[i] = lines[i].replace('U', '-')
                    lines[i] = lines[i].replace('O', '-')

            
            with open(input_path, 'w') as f:
                f.writelines(lines)
                
# if __name__ == "__main__":
#     input_dirs = ['model/input_seqs', 'model/msa_raw']
    
#     for dir_path in input_dirs:
#         print(f"\nProcessing files in {dir_path}")
#         process_files(dir_path)
