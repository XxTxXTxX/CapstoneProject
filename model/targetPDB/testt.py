from Bio import PDB
from Bio.SeqUtils import seq1
from Bio.Align import PairwiseAligner
import numpy as np
import torch

def extract_all_chains_sequences(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    chain_sequences = {}
    
    for model in structure:
        for chain in model:
            seq = []
            for residue in chain:
                if PDB.is_aa(residue, standard=True):
                    try:
                        seq.append(seq1(residue.get_resname()))
                    except Exception as e:
                        print(f"Extract Sequence {residue.get_resname()} getting error: {e}")
                        continue
            chain_sequences[chain.id] = "".join(seq)
    
    final = ""
    for chain_id, seq in chain_sequences.items():
        final += seq
    return chain_sequences, final


pdb_file = "1A04.pdb" 
_, final = extract_all_chains_sequences(pdb_file=pdb_file)
print(len(final))

def get_aligned_sequences(input_seq, pdb_seq):
    """
    Args:
        input_seq (str): Original Fasta sequence
        pdb_seq (str): PDB sequence

    Returns:
        tuple: (aligned_input_seq, aligned_pdb_seq, mask)
            - aligned_input_seq (str)
            - aligned_pdb_seq (str)
            - mask (torch.Tensor) ---> MASKED position
    """
    aligner = PairwiseAligner()
    aligner.mode = 'global' 
    aligner.open_gap_score = -2  # gap penalty
    aligner.extend_gap_score = -0.5  
    alignment = aligner.align(input_seq, pdb_seq)[0]

    aligned_input = str(alignment[0])
    aligned_pdb = str(alignment[1])

    mask = np.array([1 if res_pdb != '-' else 0 for res_pdb in aligned_pdb])

    return aligned_input, aligned_pdb, torch.tensor(mask, dtype=torch.uint8)

# input_seq = "MASMTGGQQMGRIPGNSPRMVLLESEQFLTELTRLFQKCRSSGSVFITLKKYDGRTKPIPRKSSVEGLEPAENKCLLRATDGKRKISTVVSSKEVNKFQMAYSNLLRANMDGLKKRDKKNKSKKSKPAQGGEQKLISEEDDSAGSPMPQFQTWEEFSRAAEKLYLADPMKVRVVLKYRHVDGNLCIKVTDDLVCLVYRTDQAQDVKKIEKFHSQLMRLMVAKESRNVTMETE"

# pdb_seq = final

# aligned_input, aligned_pdb, mask = get_aligned_sequences(input_seq, pdb_seq)

# Define the 37 atom types in the standard order
ATOM_TYPES = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG",
    "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "CZ", "CZ2", "CZ3", "NZ", "OXT", "OH", "TYR_OH"
]
ATOM_TYPE_INDEX = {atom: idx for idx, atom in enumerate(ATOM_TYPES)}

from Bio import PDB
import numpy as np

def extract_residue_coordinates(pdb_file, include_hetatm=False):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residue_coordinates = []
    residue_masks = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if not PDB.is_aa(residue, standard=True) and not include_hetatm:
                    continue  # Skip HETATM unless specified

                coord_tensor = torch.zeros((37, 3))
                mask_tensor = torch.zeros(37, dtype=torch.uint8)

                for atom in residue.get_atoms():
                    atom_name = atom.get_name()
                    if atom_name in ATOM_TYPE_INDEX:  
                        idx = ATOM_TYPE_INDEX[atom_name]
                        coord_tensor[idx] = torch.tensor(atom.get_coord(), dtype=torch.float32)
                        mask_tensor[idx] = 1 

                residue_coordinates.append(coord_tensor)
                residue_masks.append(mask_tensor)

    coords_tensor = torch.stack(residue_coordinates)
    mask_tensor = torch.stack(residue_masks)

    return coords_tensor, mask_tensor

# Example usage:
coords_tensor, mask_tensor = extract_residue_coordinates("1A07.pdb", include_hetatm=True)
pdb_file = "1A07.pdb"  # Replace with your actual PDB file path
coords_tensor, _ = extract_residue_coordinates(pdb_file)

print(coords_tensor.shape)

