import math

def parse_pdb(file_path):
    coordinates = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):  # 确保只处理 ATOM 行
                print(list(line.split()))
                x = float(list(line.split())[6])  # 提取 x 坐标
                y = float(list(line.split())[7])  # 提取 y 坐标
                z = float(list(line.split())[8])  # 提取 z 坐标
                coordinates.append((x, y, z))
    return coordinates

def calculate_distance(coord1, coord2):
    """
    计算两个坐标之间的欧几里得距离。
    """
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

def compare_pdb_files(file1, file2):
    """
    比较两个 PDB 文件的坐标差异。
    """
    # 解析两个文件的坐标
    coords_file1 = parse_pdb(file1)
    coords_file2 = parse_pdb(file2)

    # 确保两个文件的行数相同
    if len(coords_file1) != len(coords_file2):
        raise ValueError("两个文件的行数不同，无法逐行比较！")

    # 计算每一行的坐标差异
    differences = []
    for i, (coord1, coord2) in enumerate(zip(coords_file1, coords_file2)):
        distance = calculate_distance(coord1, coord2)
        differences.append((i + 1, distance))  # 记录行号和距离

    return differences

# 主程序
file1 = "test.pdb"  # 原始文件路径
file2 = "spread_out_structure.pdb"  # 预测文件路径

try:
    differences = compare_pdb_files(file1, file2)
    print("行号\t坐标差异")
    for line_number, distance in differences:
        print(f"{line_number}\t{distance:.4f}")
except Exception as e:
    print(f"发生错误: {e}")