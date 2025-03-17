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
    """Extracts the amino acid sequence from a PDB file."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    sequence = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    sequence += seq1(residue.get_resname())
    return sequence


def align_sequences(fasta_seq, pdb_seq):
    """
    Aligns the FASTA sequence with the PDB sequence while keeping the length of the longer sequence.
    
    Returns:
        aligned_fasta (str): Aligned FASTA sequence.
        aligned_pdb (str): Aligned PDB sequence.
        mask_fasta (torch.Tensor): 1 for matched positions in FASTA, 0 for unmatched.
        mask_pdb (torch.Tensor): 1 for matched positions in PDB, 0 for unmatched.
    """
    max_len = max(len(fasta_seq), len(pdb_seq))  # Ensure the final length matches the longest sequence

    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = -10  # Strong gap penalties
    aligner.extend_gap_score = -1
    aligner.target_end_gap_score = 0
    aligner.query_end_gap_score = 0

    alignment = aligner.align(fasta_seq, pdb_seq)[0]
    aligned_fasta = str(alignment[0]).ljust(max_len, "-")
    aligned_pdb = str(alignment[1]).ljust(max_len, "-")

    # Generate masks: 1 for real residues, 0 for gaps
    mask_fasta = torch.tensor([1 if res_fasta != '-' else 0 for res_fasta in aligned_fasta], dtype=torch.uint8)
    mask_pdb = torch.tensor([1 if res_pdb != '-' else 0 for res_pdb in aligned_pdb], dtype=torch.uint8)

    return aligned_fasta, aligned_pdb, mask_fasta, mask_pdb


def extract_residue_coordinates(pdb_file):
    """Extracts residue 3D coordinates from the PDB file."""
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


def create_final_tensor(fasta_seq, pdb_seq, pdb_coords):
    """
    Creates a tensor of shape (fasta_seq_len, 37, 3), where matched residues take values from PDB coordinates,
    and unmatched residues are filled with (0,0,0).

    Returns:
        final_tensor (torch.Tensor): Final tensor of shape (fasta_seq_len, 37, 3).
    """
    aligned_fasta, aligned_pdb, mask_fasta, mask_pdb = align_sequences(fasta_seq, pdb_seq)

    fasta_len = len(fasta_seq)
    final_tensor = torch.zeros((fasta_len, 37, 3), dtype=torch.float32)  # Initialize full tensor with zeros

    pdb_index = 0
    for i in range(fasta_len):
        if mask_fasta[i] == 1 and mask_pdb[i] == 1:
            # Copy matched residue coordinates
            final_tensor[i] = pdb_coords[pdb_index]
            pdb_index += 1

    return final_tensor


# -------------------- Example Usage --------------------

# Example FASTA sequence
fasta_seq = "MASMTGGQQMGRIPGNSPRMVLLESEQFLTELTRLFQKCRSSGSVFITLKKYDGRTKPIPRKSSVEGLEPAENKCLLRATDGKRKISTVVSSKEVNKFQMAYSNLLRANMDGLKKRDKKNKSKKSKPAQGGEQKLISEEDDSAGSPMPQFQTWEEFSRAAEKLYLADPMKVRVVLKYRHVDGNLCIKVTDDLVCLVYRTDQAQDVKKIEKFHSQLMRLMVAKESRNVTMETE"

# Load PDB file
pdb_file = "1914.pdb"
pdb_seq = extract_pdb_sequence(pdb_file)
pdb_coords = extract_residue_coordinates(pdb_file)

# Generate final tensor
final_tensor = create_final_tensor(fasta_seq, pdb_seq, pdb_coords)

# Output final tensor shape
print("Final Tensor Shape:", final_tensor.shape)