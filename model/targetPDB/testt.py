from Bio import PDB
from Bio.SeqUtils import seq1

def extract_all_chains_sequences(pdb_file):
    """
    从PDB文件中提取所有链的氨基酸序列（1-letter格式）。

    Returns:
        dict: {chain_id: sequence (str)}
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    chain_sequences = {}  # 存储每条链的氨基酸序列
    
    for model in structure:  # 遍历模型（通常只有一个）
        for chain in model:  # 遍历链
            seq = []
            for residue in chain:
                # 仅处理标准氨基酸残基
                if PDB.is_aa(residue, standard=True):
                    try:
                        seq.append(seq1(residue.get_resname()))
                    except Exception as e:
                        print(f"转换残基 {residue.get_resname()} 时出错: {e}")
                        continue
            chain_sequences[chain.id] = "".join(seq)
    
    return chain_sequences

# 示例用法：
pdb_file = "1A0R.pdb"
chain_sequences = extract_all_chains_sequences(pdb_file)

# 打印所有链的氨基酸序列
#for chain_id, sequence in chain_sequences.items():
    #print(f"链 {chain_id}: {sequence}")
final = ""
for chain_id, seq in chain_sequences.items():
    final += seq
#print(len(final))

from Bio.Align import PairwiseAligner
import numpy as np
import torch

def get_aligned_sequences(input_seq, pdb_seq):
    """
    对齐输入序列（FASTA）和PDB提取的序列，并生成 0/1 掩码，指示PDB中缺失的氨基酸残基。

    Args:
        input_seq (str): 原始FASTA输入序列
        pdb_seq (str): 从PDB文件提取的氨基酸序列

    Returns:
        tuple: (aligned_input_seq, aligned_pdb_seq, mask)
            - aligned_input_seq (str): 对齐后的输入序列（带gap）
            - aligned_pdb_seq (str): 对齐后的PDB序列（带gap）
            - mask (torch.Tensor): 0/1 掩码，形状 (L,)，1 代表有3D坐标，0 代表缺失
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'  # 使用全局对齐
    aligner.open_gap_score = -2  # 适当的gap penalty减少多余cut
    aligner.extend_gap_score = -0.5  # 允许更平滑的gap扩展
    alignment = aligner.align(input_seq, pdb_seq)[0]  # 选取最高分的对齐结果

    # 获取对齐后的序列
    aligned_input = str(alignment[0])
    aligned_pdb = str(alignment[1])

    # 生成0/1掩码
    mask = np.array([1 if res_pdb != '-' else 0 for res_pdb in aligned_pdb])

    return aligned_input, aligned_pdb, torch.tensor(mask, dtype=torch.uint8)

# 示例用法：
input_seq = "XSELDQLRQEAEQLKNQIRDARKACADATLSQITNNIDPVGRIQMRTRRTLRGHLAKIYAMHWGTDSRLLLSASQDGKLIIWDSYTTNKVHAIPLRSSWVMTCAYAPSGNYVACGGLDNICSIYNLKTREGNVRVSRELAGHTGYLSCCRFLDDNQIVTSSGDTTCALWDIETGQQTTTFTGHTGDVMSLSLAPDTRLFVSGACDASAKLWDVREGMCRQTFTGHESDINAICFFPNGNAFATGSDDATCRLFDLRADQELMTYSHDNIICGITSVSFSKSGRLLLAGYDDFNCNVWDALKADRAGVLAGHDNRVSCLGVTDDGMAVATGSWDSFLKIWNPVINIEDLTEKDKLKMEVDQLKKEVTLERMLVSKCCEEFRDYVEERSGEDPLVKGIPEDKNPFKEMEKAKSQSLEEDFEGQASHTGPKGVINDWRKFKLESEDSDSVAHSKKEILRQMSSPQSRDDKDSKERFSRKMSVQEYELIHKDKEDENCLRKYRRQCMQDMHQKLSFGPRYGFVYELESGEQFLETIEKEQKITTIVVHIYEDGIKGCDALNSSLICLAAEYPMVKFCKIKASNTGAGDRFSSDVLPTLLVYKGGELLSNFISVTEQLAEEFFTGDVESFLNEYGLLPEKEMHVLEQTNMEEDME"
pdb_seq = final  # PDB提取的序列

aligned_input, aligned_pdb, mask = get_aligned_sequences(input_seq, pdb_seq)

# 打印修正后的对齐结果
print("修正后的对齐输入序列:\n", aligned_input)
print("修正后的对齐PDB序列:\n", aligned_pdb)
print("掩码 (1=有坐标, 0=缺失):\n", mask)
# 示例用法：
input_seq = "XSELDQLRQEAEQLKNQIRDARKACADATLSQITNNIDPVGRIQMRTRRTLRGHLAKIYAMHWGTDSRLLLSASQDGKLIIWDSYTTNKVHAIPLRSSWVMTCAYAPSGNYVACGGLDNICSIYNLKTREGNVRVSRELAGHTGYLSCCRFLDDNQIVTSSGDTTCALWDIETGQQTTTFTGHTGDVMSLSLAPDTRLFVSGACDASAKLWDVREGMCRQTFTGHESDINAICFFPNGNAFATGSDDATCRLFDLRADQELMTYSHDNIICGITSVSFSKSGRLLLAGYDDFNCNVWDALKADRAGVLAGHDNRVSCLGVTDDGMAVATGSWDSFLKIWNPVINIEDLTEKDKLKMEVDQLKKEVTLERMLVSKCCEEFRDYVEERSGEDPLVKGIPEDKNPFKEMEKAKSQSLEEDFEGQASHTGPKGVINDWRKFKLESEDSDSVAHSKKEILRQMSSPQSRDDKDSKERFSRKMSVQEYELIHKDKEDENCLRKYRRQCMQDMHQKLSFGPRYGFVYELESGEQFLETIEKEQKITTIVVHIYEDGIKGCDALNSSLICLAAEYPMVKFCKIKASNTGAGDRFSSDVLPTLLVYKGGELLSNFISVTEQLAEEFFTGDVESFLNEYGLLPEKEMHVLEQTNMEEDME"
pdb_seq = final
aligned_input, aligned_pdb, mask = get_aligned_sequences(input_seq, pdb_seq)

# Nres, 37, 3 [[],[],[], ...]


# 打印对齐结果
#print("对齐的输入序列:\n", aligned_input)
#print("对齐的PDB序列:\n", aligned_pdb)
#print("掩码 (1=有坐标, 0=缺失):\n", mask)
#print(f"mask len: {len(mask)}")

from Bio import PDB
import numpy as np

def extract_residue_coordinates(pdb_file):
    """
    Extracts 3D coordinates for all residues (including HETATM and ATOM records) from a PDB file.

    Returns:
        dict: {residue_id: np.array of shape (num_atoms, 3)}
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residue_coordinates = {}

    for model in structure:  # Iterate through models (usually just one)
        for chain in model:  # Iterate through chains
            for residue in chain:  # Iterate through residues
                res_id = residue.get_id()
                res_key = f"{chain.id}_{res_id[1]}_{residue.get_resname()}"  # Unique ID: chain_resnum_resname
                atom_coords = []

                for atom in residue.get_atoms():
                    atom_coords.append(atom.get_coord())  # Extract XYZ coordinates
                
                if atom_coords:
                    residue_coordinates[res_key] = np.array(atom_coords)  # Convert to NumPy array

    return residue_coordinates

# Example usage:
pdb_file = "1914.pdb"  # Replace with your actual PDB file path
residue_coords = extract_residue_coordinates(pdb_file)

# Print first 5 residues for inspection
for res, coords in list(residue_coords.items())[:-1]:
    print(f"Residue: {res}, Coordinates:\n{coords}\n")