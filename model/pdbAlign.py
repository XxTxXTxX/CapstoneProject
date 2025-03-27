from Bio import PDB
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner
import torch
import numpy as np

# Define standard atom types
ATOM_TYPES = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "CZ", "CZ2", "CZ3", "NZ", "OXT", "OH", "TYR_OH"
]
ATOM_TYPE_INDEX = {atom: idx for idx, atom in enumerate(ATOM_TYPES)}


def extract_pdb_sequence(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    sequence += seq1(residue.get_resname())
    return sequence



def align_sequences(seq1, seq2, match_score=1, mismatch_score=-1, gap_penalty=-2):
    m, n = len(seq1), len(seq2)

    # **1. Initialize DP matrix and traceback matrix**
    dp = np.zeros((m + 1, n + 1))
    traceback = np.zeros((m + 1, n + 1), dtype=int)  # 0 = diagonal, 1 = up, 2 = left

    # **2. Initialize first row and column with gap penalties**
    for i in range(1, m + 1):
        dp[i][0] = dp[i-1][0] + gap_penalty
        traceback[i][0] = 1  # Up direction (gap in seq2)
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j-1] + gap_penalty
        traceback[0][j] = 2  # Left direction (gap in seq1)

    # **3. Fill DP matrix**
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i-1][j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_score)
            delete = dp[i-1][j] + gap_penalty  # Gap in seq2
            insert = dp[i][j-1] + gap_penalty  # Gap in seq1

            dp[i][j] = max(match, delete, insert)
            traceback[i][j] = np.argmax([match, delete, insert])  # Store direction

    # **4. Backtrack to get alignment**
    aligned_seq1, aligned_seq2 = [], []
    mask_seq1, mask_seq2 = [], []

    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and traceback[i][j] == 0:  # Diagonal (match/mismatch)
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append(seq2[j-1])
            mask_seq1.append(1)
            mask_seq2.append(1)
            i -= 1
            j -= 1
        elif i > 0 and traceback[i][j] == 1:  # Up (gap in seq2)
            aligned_seq1.append(seq1[i-1])
            aligned_seq2.append('-')
            mask_seq1.append(1)
            mask_seq2.append(0)
            i -= 1
        else:  # Left (gap in seq1)
            aligned_seq1.append('-')
            aligned_seq2.append(seq2[j-1])
            mask_seq1.append(0)
            mask_seq2.append(1)
            j -= 1

    # Reverse the sequences (since we built them backward)
    aligned_seq1 = ''.join(aligned_seq1[::-1])
    aligned_seq2 = ''.join(aligned_seq2[::-1])
    mask_seq1 = np.array(mask_seq1[::-1])
    mask_seq2 = np.array(mask_seq2[::-1])

    assert len(aligned_seq2) == len(aligned_seq1)

    li = []
    pdb, fasta = 0, 0
    pdb_idx = []
    fasta_idx = []
    i = 0
    aligned_fasta = aligned_seq1
    aligned_pdb = aligned_seq2
    
    while (i < len(aligned_pdb)):
      if aligned_pdb[i] == aligned_fasta[i]:
        li.append(aligned_fasta[i])
        fasta_idx.append(fasta)
        pdb_idx.append(pdb)
      if aligned_fasta[i] != "-":
        fasta += 1
      if aligned_pdb[i] != "-":
        pdb += 1
      i += 1

    return pdb_idx, fasta_idx


def extract_residue_coordinates(pdb_file): # MASK
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residue_coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not PDB.is_aa(residue, standard=True):
                    continue  # Ignore HETATM

                coord_tensor = torch.zeros((37, 3))  # Initialize with zeros
                for atom in residue.get_atoms():
                    atom_name = atom.get_name()
                    if atom_name in ATOM_TYPE_INDEX:
                        idx = ATOM_TYPE_INDEX[atom_name]
                        coord_tensor[idx] = torch.tensor(atom.get_coord(), dtype=torch.float32)
                residue_coordinates.append(coord_tensor)

    return residue_coordinates


def create_final_tensor(fasta_seq, pdb_coords, fasta_idx, pdb_idx):
    fasta_length = len(fasta_seq)
    final_tensor = torch.zeros((fasta_length, 37, 3), dtype=torch.float32)  # Default to zeros

    # Map the PDB coordinates to the matching FASTA indices
    for fasta_position, pdb_position in zip(fasta_idx, pdb_idx):
        final_tensor[fasta_position] = pdb_coords[pdb_position]

    return final_tensor