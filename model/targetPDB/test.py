import torch
from Bio import PDB
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner

import numpy as np

#123456789
#123678
def extract_all_chains_sequence_and_coords(pdb_file):
    """
    从PDB文件中提取所有链的序列和CA原子坐标。
    
    返回一个字典，键为链ID，值为元组 (sequence, coords)，
    其中 sequence 为该链的氨基酸序列，coords 为每个残基的CA坐标列表（缺失则为None）。
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    chain_data = {}
    # 遍历所有模型和链（大多数PDB只包含一个model）
    for model in structure:
        for chain in model:
            seq = ""
            coords = []
            for residue in chain:
                # 仅处理标准氨基酸残基
                if PDB.is_aa(residue, standard=True):
                    try:
                        # 转换三字母代码为一字母代码
                        seq += seq1(residue.get_resname())
                    except Exception as e:
                        print(f"转换残基 {residue.get_resname()} 时出错: {e}")
                        continue
                    # 这里示例使用CA原子，你可以根据需要修改
                    if 'CA' in residue:
                        coords.append(residue['CA'].get_coord())
                    else:
                        coords.append(None)
            chain_data[chain.id] = (seq, coords)
    return chain_data

def print_chain_info(chain_data):
    """打印每个链的基本信息"""
    for chain_id, (seq, coords) in chain_data.items():
        print(f"链 {chain_id}:")
        print(f"  序列长度: {len(seq)}")
        valid_coords = sum(1 for c in coords if c is not None)
        print(f"  含结构数据的残基数: {valid_coords}")

def get_aligned_sequences(aln, seq1, seq2):
    """
    根据 Alignment 对象的 aligned 属性，构造带 gap 的对齐序列。
    注意：此实现基于 aligned 属性可能只有一组连续匹配块，
    对于复杂对齐可能需要更完善的处理。
    """
    aligned_ranges1, aligned_ranges2 = aln.aligned  # 分别是 seq1 和 seq2 的匹配区域列表
    aligned_seq1 = []
    aligned_seq2 = []
    prev1, prev2 = 0, 0
    for (start1, end1), (start2, end2) in zip(aligned_ranges1, aligned_ranges2):
        # 填充前面的gap
        if start1 > prev1:
            aligned_seq1.append(seq1[prev1:start1])
            aligned_seq2.append('-' * (start1 - prev1))
        if start2 > prev2:
            aligned_seq1.append('-' * (start2 - prev2))
            aligned_seq2.append(seq2[prev2:start2])
        # 添加对齐块
        aligned_seq1.append(seq1[start1:end1])
        aligned_seq2.append(seq2[start2:end2])
        prev1, prev2 = end1, end2
    # 补全尾部
    if prev1 < len(seq1):
        aligned_seq1.append(seq1[prev1:])
        aligned_seq2.append('-' * (len(seq1) - prev1))
    if prev2 < len(seq2):
        aligned_seq1.append('-' * (len(seq2) - prev2))
        aligned_seq2.append(seq2[prev2:])
    return "".join(aligned_seq1), "".join(aligned_seq2)

def align_and_build_tensor(input_seq, pdb_seq, pdb_coords, N_atoms=1):
    aligner = PairwiseAligner()
    alignment = aligner.align(input_seq, pdb_seq)[0]
    # 获取带 gap 的对齐序列
    aln_input, aln_pdb = get_aligned_sequences(alignment, input_seq, pdb_seq)
    
    # 初始化 tensor，这里 tensor 的长度仍然基于原始 input_seq 长度
    len_input = len(input_seq)
    coords_tensor = np.zeros((len_input, N_atoms, 3))
    mask = np.zeros((len_input, N_atoms))
    
    pdb_index = 0
    input_index = 0
    # 遍历对齐结果，注意两条序列长度相同
    for i in range(len(aln_input)):
        if aln_input[i] != '-' and aln_pdb[i] != '-':
            if pdb_index < len(pdb_coords) and pdb_coords[pdb_index] is not None:
                coords_tensor[input_index, 0] = pdb_coords[pdb_index]
                mask[input_index, 0] = 1
            pdb_index += 1
            input_index += 1
        elif aln_input[i] != '-' and aln_pdb[i] == '-':
            input_index += 1
        elif aln_input[i] == '-' and aln_pdb[i] != '-':
            pdb_index += 1
    return torch.tensor(coords_tensor, dtype=torch.float32), torch.tensor(mask, dtype=torch.bool)

if __name__ == "__main__":
    # 假设输入序列为某个长度为100的序列
    input_seq = "XSELDQLRQEAEQLKNQIRDARKACADATLSQITNNIDPVGRIQMRTRRTLRGHLAKIYAMHWGTDSRLLLSASQDGKLIIWDSYTTNKVHAIPLRSSWVMTCAYAPSGNYVACGGLDNICSIYNLKTREGNVRVSRELAGHTGYLSCCRFLDDNQIVTSSGDTTCALWDIETGQQTTTFTGHTGDVMSLSLAPDTRLFVSGACDASAKLWDVREGMCRQTFTGHESDINAICFFPNGNAFATGSDDATCRLFDLRADQELMTYSHDNIICGITSVSFSKSGRLLLAGYDDFNCNVWDALKADRAGVLAGHDNRVSCLGVTDDGMAVATGSWDSFLKIWNPVINIEDLTEKDKLKMEVDQLKKEVTLERMLVSKCCEEFRDYVEERSGEDPLVKGIPEDKNPFKEMEKAKSQSLEEDFEGQASHTGPKGVINDWRKFKLESEDSDSVAHSKKEILRQMSSPQSRDDKDSKERFSRKMSVQEYELIHKDKEDENCLRKYRRQCMQDMHQKLSFGPRYGFVYELESGEQFLETIEKEQKITTIVVHIYEDGIKGCDALNSSLICLAAEYPMVKFCKIKASNTGAGDRFSSDVLPTLLVYKGGELLSNFISVTEQLAEEFFTGDVESFLNEYGLLPEKEMHVLEQTNMEEDME" # 示例序列
    # 提取PDB文件中链的信息（假设链 'A'）
    pdb_file = "1A0R.pdb"
    chain_data = extract_all_chains_sequence_and_coords(pdb_file)
    print_chain_info(chain_data)
    
    coords, mask = align_and_build_tensor(input_seq, pdb_seq, pdb_coords, N_atoms=1)
    
    #print(f"坐标张量形状: {coords.shape}")
    #print(f"Mask张量形状: {mask.shape}")
    #print(f"每个残基的有结构信息数: {mask.float().sum(dim=1)}")
