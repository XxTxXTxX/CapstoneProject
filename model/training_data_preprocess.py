import os
resType = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V","X","-"]
def process_files(input_dir, batch_size=100):
    file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.a3m') or f.endswith('.fasta')]
    
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        print(f"Processing batch {i // batch_size + 1} ({len(batch)} files)")

        for input_path in batch:
            try:
                temp_output = []
                with open(input_path, 'r') as f: 
                    for i, line in enumerate(f):
                        if i % 2 == 1:
                            line = line.replace('B', '-').replace('Z', '-').replace('U', '-').replace('O', '-')
                        temp_output.append(line)

                with open(input_path, 'w') as f:
                    f.writelines(temp_output)

                print(f"Processed: {input_path}")

            except Exception as e:
                print(f"Error processing {input_path}: {e}")

# if __name__ == "__main__":
#     input_dirs = ['model/input_seqs', 'model/msa_raw']
    
#     for dir_path in input_dirs:
#         print(f"\nProcessing files in {dir_path}")
#         process_files(dir_path)
