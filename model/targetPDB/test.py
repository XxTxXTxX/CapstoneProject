import torch
from Bio import PDB
import numpy as np

def extract_atom_coordinates_37(pdb_file, chain_order=['B', 'G', 'P']):
    """
    从PDB文件中提取原子坐标，转换为[N_res, 37, 3]格式
    
    Args:
        pdb_file: PDB文件路径
        chain_order: 链的顺序列表
    
    Returns:
        coordinates: shape [N_res, 37, 3] 的tensor
        atom_mask: shape [N_res, 37] 的mask tensor
    """
    # 初始化输出数组
    N_res = 650  # 总残基数
    N_atoms = 37  # 每个残基的原子数
    coords = np.zeros((N_res, N_atoms, 3))
    mask = np.zeros((N_res, N_atoms))
    
    # 记录当前处理的残基索引
    residue_counter = 0
    
    # 使用BioPython解析PDB文件
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # 遍历每条链
    for chain_id in chain_order:
        chain = structure[0][chain_id]  # 假设只有一个model
        for residue in chain:
            if residue_counter >= N_res:
                break
                
            # 获取残基中的所有原子
            atom_dict = {atom.name: atom for atom in residue}
            
            # 填充坐标和mask
            for atom_idx, atom_name in enumerate(atom_dict.keys()):
                if atom_idx >= N_atoms:
                    break
                    
                atom = atom_dict[atom_name]
                coords[residue_counter, atom_idx] = atom.get_coord()
                mask[residue_counter, atom_idx] = 1
                
            residue_counter += 1
    
    return (torch.tensor(coords, dtype=torch.float32), 
            torch.tensor(mask, dtype=torch.bool))

def verify_coordinates(coords, mask):
    """验证提取的坐标"""
    print(f"坐标张量形状: {coords.shape}")
    print(f"Mask张量形状: {mask.shape}")
    print(f"每个残基的平均原子数: {mask.float().sum(dim=1).mean()}")
    print(f"非零坐标数: {(coords != 0).sum().item()}")

# 使用示例
if __name__ == "__main__":
    pdb_file = "model/targetPDB/1A0R.pdb"
    coords, mask = extract_atom_coordinates_37(pdb_file)
    verify_coordinates(coords, mask)
    
    # 检查第一个残基的原子
    print("\n第一个残基的原子坐标:")
    for i in range(37):
        if mask[0, i]:
            print(f"原子 {i}: {coords[0, i]}")