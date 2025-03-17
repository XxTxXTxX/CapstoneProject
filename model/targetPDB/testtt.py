from Bio import PDB
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner
import numpy as np
import torch

# Define the 37 atom types in the standard order
ATOM_TYPES = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "CZ", "CZ2", "CZ3", "NZ", "OXT", "OH", "TYR_OH"
]
ATOM_TYPE_INDEX = {atom: idx for idx, atom in enumerate(ATOM_TYPES)}

# -------------------- 1. Extract Sequence from PDB --------------------
def extract_pdb_sequence(pdb_file):
    """
    Extracts one-letter amino acid sequence from a PDB file.
    Returns:
        str: The concatenated amino acid sequence from all chains.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    sequence += seq1(residue.get_resname())

    return sequence


# -------------------- 2. Align FASTA and PDB Sequences --------------------
def align_sequences(fasta_seq, pdb_seq):
    """
    Aligns the FASTA sequence with the PDB sequence using dynamic programming.

    Args:
        fasta_seq (str): The reference sequence from the FASTA file.
        pdb_seq (str): The extracted sequence from the PDB file.

    Returns:
        tuple: (aligned_fasta, aligned_pdb, mask) where mask is 1 for existing residues and 0 for gaps.
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(fasta_seq, pdb_seq)[0]

    aligned_fasta = str(alignment[0])
    aligned_pdb = str(alignment[1])

    mask = np.array([1 if res_pdb != '-' else 0 for res_pdb in aligned_pdb])

    return aligned_fasta, aligned_pdb, torch.tensor(mask, dtype=torch.uint8)


# -------------------- 3. Extract 3D Coordinates --------------------
def extract_residue_coordinates(pdb_file):
    """
    Extracts 3D coordinates from a PDB file and stores them in a tensor.

    Returns:
        torch.Tensor: Tensor of shape (Nres, 37, 3), aligned to the input FASTA sequence.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residue_coordinates = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if not PDB.is_aa(residue, standard=True):
                    continue  # Skip HETATM

                coord_tensor = torch.zeros((37, 3))  # Initialize with zeros

                for atom in residue.get_atoms():
                    atom_name = atom.get_name()
                    if atom_name in ATOM_TYPE_INDEX:
                        idx = ATOM_TYPE_INDEX[atom_name]
                        coord_tensor[idx] = torch.tensor(atom.get_coord(), dtype=torch.float32)

                residue_coordinates.append(coord_tensor)

    return residue_coordinates


# -------------------- 4. Fix Length Discrepancy with DP --------------------
def adjust_pdb_coordinates(fasta_seq, pdb_seq, pdb_coords):
    """
    Adjusts the PDB coordinate tensor to match the length of the input FASTA sequence.
    If PDB has extra residues, it removes them optimally.
    If PDB is missing residues, it pads missing positions with (0,0,0).

    Args:
        fasta_seq (str): The original FASTA sequence.
        pdb_seq (str): The aligned PDB sequence.
        pdb_coords (list of tensors): List of coordinate tensors from PDB.

    Returns:
        torch.Tensor: Adjusted coordinates of shape (len(fasta_seq), 37, 3).
    """
    aligned_fasta, aligned_pdb, mask = align_sequences(fasta_seq, pdb_seq)

    adjusted_coords = []
    pdb_index = 0

    for i, aa in enumerate(aligned_fasta):
        if aa == "-":
            # If FASTA has a gap but PDB does not, skip it.
            continue

        if aligned_pdb[i] == "-":
            # If PDB has a gap, insert (0,0,0) for all atoms.
            adjusted_coords.append(torch.zeros((37, 3), dtype=torch.float32))
        else:
            # If both sequences match, take the PDB coordinate.
            adjusted_coords.append(pdb_coords[pdb_index])
            pdb_index += 1

    return torch.stack(adjusted_coords), mask


# -------------------- 5. Main Execution --------------------
def process_pdb(fasta_seq, pdb_file):
    """
    Processes a PDB file and aligns its residues to a given FASTA sequence.

    Args:
        fasta_seq (str): The sequence from the FASTA file.
        pdb_file (str): Path to the PDB file.

    Returns:
        torch.Tensor: Final tensor of shape (len(fasta_seq), 37, 3).
    """
    pdb_seq = extract_pdb_sequence(pdb_file)
    pdb_coords = extract_residue_coordinates(pdb_file)

    adjusted_coords, mask = adjust_pdb_coordinates(fasta_seq, pdb_seq, pdb_coords)
    return adjusted_coords


# # -------------------- TEST --------------------
# fasta_seq = "SNQEPATILLIDDHPMLRTGVKQLISMAPDITVVGEASNGEQGIELAESLDPDLILLDLNMPGMNGLETLDKLREKSLSGRIVVFSVSNHEEDVVTALKRGADGYLLKDMEPEDLLKALHQAAAGEMVLSEALTPVLAASLRANRATTERDVNQLTPRERDILKLIAQGLPNKMIARRLDITESTVKVHVKHMLKKMKLKSRVEAAVWVHQERIF"  # Example sequence
# pdb_file = "1A04.pdb"

# final_coords, mask = process_pdb(fasta_seq, pdb_file)
# #print("Final Coordinate Tensor Shape:", final_coords.shape)
# print(final_coords[-1])
# #print(mask)
# # test 2

# fasta_seq = "MASMTGGQQMGRIPGNSPRMVLLESEQFLTELTRLFQKCRSSGSVFITLKKYDGRTKPIPRKSSVEGLEPAENKCLLRATDGKRKISTVVSSKEVNKFQMAYSNLLRANMDGLKKRDKKNKSKKSKPAQGGEQKLISEEDDSAGSPMPQFQTWEEFSRAAEKLYLADPMKVRVVLKYRHVDGNLCIKVTDDLVCLVYRTDQAQDVKKIEKFHSQLMRLMVAKESRNVTMETE"  # Example sequence
# pdb_file = "1914.pdb"

# final_coords, mask = process_pdb(fasta_seq, pdb_file)
# print("Final Coordinate Tensor Shape:", final_coords.shape)
