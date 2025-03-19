import os
resType = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V","X","-"]
def process_files(input_dir, output_dir=None):
    """
    处理指定目录下的所有文件，将双数行中的'Z'替换为'-'
    
    Args:
        input_dir: 输入文件夹路径
        output_dir: 输出文件夹路径（如果不指定，则直接修改原文件）
    """
    
    # 处理目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.a3m') or filename.endswith('.fasta'): 
            input_path = os.path.join(input_dir, filename)
            
            # 读取文件内容
            with open(input_path, 'r') as f:
                lines = f.readlines()
            
            # 处理每一行
            for i in range(len(lines)):
                if i % 2 == 1:  # 双数行
                    lines[i] = lines[i].replace('B', '-')
                    lines[i] = lines[i].replace('Z', '-')
                    lines[i] = lines[i].replace('U', '-')

            
            # 写入处理后的内容
            with open(input_path, 'w') as f:
                f.writelines(lines)
            
            # print(f"Processed {filename}")

if __name__ == "__main__":
    input_dirs = ['model/input_seqs', 'model/msa_raw']
    
    for dir_path in input_dirs:
        print(f"\nProcessing files in {dir_path}")
        process_files(dir_path)