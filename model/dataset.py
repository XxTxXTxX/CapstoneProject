import torch
import torch.nn as nn
from torch.utils.data import Dataset
from model import ModelArgs
from Bio import PDB
import targetPDB.testtt as tt
import re
import os


class ProcessDataset(Dataset):
    def __init__(self, temp_Ph_vals):
        self.msa_file_path = "model/msa_raw"
        self.pdb_file_path = "model/targetPDB"
        self.ATOM_TYPES = ["N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "OG", "OG1", "SG",
                           "CD", "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
                           "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "CZ", "CZ2", "CZ3", "NZ", "OXT", "OH", "TYR_OH"
                          ]
        self.ATOM_TYPE_INDEX = {atom: idx for idx, atom in enumerate(self.ATOM_TYPES)}
        self.msa_files = [
            os.path.join(self.msa_file_path, f) 
            for f in os.listdir(self.msa_file_path) 
            if f.endswith('.a3m')
        ]
        # print(self.msa_files)
        self.feature_extractor = featureExtraction()
        self.features = []
        self.__preprocess_all_msa(temp_Ph_vals)
        for atom in self.features:
            filename = atom['seq_name']
            file = "model/input_seqs/" + filename + ".fasta"
            with open(file, 'r') as f:
                lines = f.readlines()
            seq = lines[1].strip()
            # print(seq)
            atom['coordinates'] = tt.process_pdb(seq, os.path.join(self.pdb_file_path, atom['seq_name'] + ".pdb"))

    def __preprocess_all_msa(self, temp_Ph_vals):
        for msa_path in self.msa_files:
            batch = self.feature_extractor.create_features_from_a3m(msa_path, temp_Ph_vals)
            self.features.append(batch)

    # def __extract_residue_coordinates(self, pdb_file):
    #     """
    #     Extracts 3D coordinates for all standard residues (excluding HETATM) from a PDB file
    #     and converts them into a tensor of shape (Nres, 37, 3).
        
    #     Returns:
    #         torch.Tensor: (Nres, 37, 3) where each residue has XYZ coordinates for its atoms. --> Already containing masked information
    #     """
    #     parser = PDB.PDBParser(QUIET=True)
    #     structure = parser.get_structure("protein", pdb_file)
    #     residue_list = [] 
    #     residue_coordinates = [] 
    #     residue_masks = []  

    #     for model in structure:
    #         for chain in model:
    #             for residue in chain:
    #                 if not PDB.is_aa(residue, standard=True):
    #                     continue  # Ignore HETATM
    #                 residue_list.append(f"{chain.id}_{residue.get_id()[1]}_{residue.get_resname()}")

    #                 # Initialize coordinate tensor and mask (Nres, 37, 3)
    #                 coord_tensor = torch.zeros((37, 3))  
    #                 mask_tensor = torch.zeros(37, dtype=torch.uint8) 
    #                 for atom in residue.get_atoms():
    #                     atom_name = atom.get_name()
    #                     if atom_name in self.ATOM_TYPE_INDEX:  
    #                         idx = self.ATOM_TYPE_INDEX[atom_name] 
    #                         coord_tensor[idx] = torch.tensor(atom.get_coord(), dtype=torch.float32)
    #                         mask_tensor[idx] = 1 

    #                 residue_coordinates.append(coord_tensor)
    #                 residue_masks.append(mask_tensor)

    #     coords_tensor = torch.stack(residue_coordinates)  # Shape: (Nres, 37, 3)

    #     return coords_tensor
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx]


            

class featureExtraction():

    def __init__(self):
        self._restypes = ["A","R","N","D","C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V",]
        self._restypes_with_x = self._restypes + ["X"]
        self._restypes_with_x_and_gap = self._restypes_with_x + ["-"]
        self.restype_order_with_x = None
        self.restype_order_with_x_and_gap = None
        self.restype_order_with_x = {res: i for i, res in enumerate(self._restypes_with_x)}
        self.restype_order_with_x_and_gap = {res: i for i, res in enumerate(self._restypes_with_x_and_gap)}

    def load_a3m_file(self, file_name: str):
        seqs = None             
        with open(file_name, 'r') as f:
            lines = f.readlines()
        description_line_indices = [i for i,l in enumerate(lines) if l.startswith('>')] 
        seqs = [lines[i+1].strip() for i in description_line_indices] 
        return seqs

    def onehot_encode_aa_type(self, seq, include_gap_token=False):
        restype_order = self.restype_order_with_x if not include_gap_token else self.restype_order_with_x_and_gap
        encoding = None
        sequence_inds = torch.tensor([restype_order[a] for a in seq])
        encoding = nn.functional.one_hot(sequence_inds, num_classes=len(restype_order))
        return encoding
    
    def initial_data_from_seqs(self, seqs):
        unique_seqs = None
        deletion_count_matrix = None
        aa_distribution = None

        deletion_count_matrix = []
        unique_seqs = []
        for seq in seqs:
            deletion_count_list = []
            deletion_counter = 0
            for letter in seq:
                if letter.islower():
                    deletion_counter += 1
                else:
                    deletion_count_list.append(deletion_counter)
                    deletion_counter=0
            seq_without_deletion = re.sub('[a-z]', '', seq)

            if seq_without_deletion in unique_seqs:
                continue

            unique_seqs.append(seq_without_deletion)
            deletion_count_matrix.append(deletion_count_list)
        
        unique_seqs = torch.stack([self.onehot_encode_aa_type(seq, include_gap_token=True) for seq in unique_seqs], dim=0)
        # N_seq, N_res, 22
        unique_seqs = unique_seqs.float()
        # N_seq, N_res
        deletion_count_matrix = torch.tensor(deletion_count_matrix).float()
        # N_res, 22
        aa_distribution = unique_seqs.float().mean(dim=0)

        return { 'msa_aatype': unique_seqs, 'msa_deletion_count': deletion_count_matrix, 'aa_distribution': aa_distribution}
    

    def select_cluster_centers(self, features, max_msa_clusters=512, seed=None):
        N_seq, N_res = features['msa_aatype'].shape[:2]
        MSA_FEATURE_NAMES = ['msa_aatype', 'msa_deletion_count']
        max_msa_clusters = min(max_msa_clusters, N_seq)

        gen = None
        if seed is not None:
            gen = torch.Generator(features['msa_aatype'].device)
            gen.manual_seed(seed)

        shuffled = torch.randperm(N_seq-1, generator=gen) + 1
        shuffled = torch.cat((torch.tensor([0]), shuffled), dim=0)

        for key in MSA_FEATURE_NAMES:
            extra_key = f'extra_{key}'
            value = features[key]
            features[extra_key] = value[shuffled[max_msa_clusters:]]
            features[key] = value[shuffled[:max_msa_clusters]]
        return features
    

    def mask_cluster_centers(self, features, mask_probability=0.15, seed=None):
        N_clust, N_res = features['msa_aatype'].shape[:2]
        N_aa_categories = 23
        odds = {
            'uniform_replacement': 0.1,
            'replacement_from_distribution': 0.1,
            'no_replacement': 0.1,
            'masked_out': 0.7,
        }
        gen = None
        if seed is not None:
            gen = torch.Generator(features['msa_aatype'].device)
            gen.manual_seed(seed)
            torch.manual_seed(seed)
        uniform_replacement = torch.tensor([1/20]*20+[0,0]) * odds['uniform_replacement']
        replacement_from_distribution = features['aa_distribution'] * odds['replacement_from_distribution']
        no_replacement = features['msa_aatype'] * odds['no_replacement']
        masked_out = torch.ones((N_clust, N_res, 1)) * odds['masked_out']

        uniform_replacement = uniform_replacement[None, None, ...].broadcast_to(no_replacement.shape)
        replacement_from_distribution = replacement_from_distribution[None, ...].broadcast_to(no_replacement.shape)
        categories_without_mask_token = uniform_replacement + replacement_from_distribution + no_replacement
        categories_with_mask_token = torch.cat((categories_without_mask_token, masked_out), dim=-1)
        categories_with_mask_token = categories_with_mask_token.reshape(-1, N_aa_categories)
        replace_with = torch.distributions.Categorical(categories_with_mask_token).sample()
        replace_with = nn.functional.one_hot(replace_with, num_classes=N_aa_categories)
        replace_with = replace_with.reshape(N_clust, N_res, N_aa_categories)
        replace_with = replace_with.float()
        replace_mask = torch.rand((N_clust, N_res), generator=gen) < mask_probability
        features['true_msa_aatype'] = features['msa_aatype'].clone()
        aatype_padding = torch.zeros((N_clust, N_res, 1))
        features['msa_aatype'] = torch.cat(
            (features['msa_aatype'], aatype_padding), 
            dim=-1)
        features['msa_aatype'][replace_mask] = replace_with[replace_mask]

        return features
    
    def cluster_assignment(self, features):       
        N_clust, N_res = features['msa_aatype'].shape[:2]
        N_extra = features['extra_msa_aatype'].shape[0]
        msa_aatype = features['msa_aatype'][...,:21]
        extra_msa_aatype = features['extra_msa_aatype'][...,:21]
        agreement = torch.einsum('cra,era->ce', msa_aatype, extra_msa_aatype)
        assignment = torch.argmax(agreement,dim=0)
        features['cluster_assignment'] = assignment
        assignment_counts = torch.bincount(assignment, minlength=N_clust)
        features['cluster_assignment_counts'] = assignment_counts
        return features
    
    def cluster_average(self, feature, extra_feature, cluster_assignment, cluster_assignment_count):
        N_clust, N_res = feature.shape[:2]
        N_extra = extra_feature.shape[0]
        unsqueezed_extra_shape = (N_extra,) + (1,) * (extra_feature.dim()-1)
        unsqueezed_cluster_shape = (N_clust,) + (1,) * (feature.dim()-1)
        cluster_assignment = cluster_assignment.view(unsqueezed_extra_shape).broadcast_to(extra_feature.shape)
        cluster_sum = torch.scatter_add(feature, dim=0, index=cluster_assignment, src=extra_feature)
        cluster_assignment_count = cluster_assignment_count.view(unsqueezed_cluster_shape).broadcast_to(feature.shape)
        cluster_average = cluster_sum / (cluster_assignment_count + 1)

        return cluster_average
    
    def summarize_clusters(self, features):
        N_clust, N_res = features['msa_aatype'].shape[:2]
        N_extra = features['extra_msa_aatype'].shape[0]
        cluster_deletion_mean = self.cluster_average(
            features['msa_deletion_count'],
            features['extra_msa_deletion_count'],
            features['cluster_assignment'],
            features['cluster_assignment_counts'] 
        )

        cluster_deletion_mean = 2/torch.pi * torch.arctan(cluster_deletion_mean/3)
        extra_msa_aatype = features['extra_msa_aatype']
        pad = torch.zeros(extra_msa_aatype.shape[:-1]+(1,), dtype=extra_msa_aatype.dtype, device=extra_msa_aatype.device)
        extra_msa_aatype_padded = torch.cat((extra_msa_aatype, pad), dim=-1)
        
        cluster_profile = self.cluster_average(
            features['msa_aatype'],
            extra_msa_aatype_padded,
            features['cluster_assignment'],
            features['cluster_assignment_counts']
        )

        features['cluster_deletion_mean'] = cluster_deletion_mean
        features['cluster_profile'] = cluster_profile
        return features
    
    def crop_extra_msa(self, features, max_extra_msa_count=5120, seed=None):
        N_extra = features['extra_msa_aatype'].shape[0]
        gen = None
        if seed is not None:
            gen = torch.Generator(features['extra_msa_aatype'].device)
            gen.manual_seed(seed)

        max_extra_msa_count = min(max_extra_msa_count, N_extra)
        inds_to_select = torch.randperm(N_extra, generator=gen)[:max_extra_msa_count]
        for k in features.keys():
            if k.startswith('extra_'):
                features[k] = features[k][inds_to_select]

        return features
    

    def calculate_msa_feat(self, features):
        N_clust, N_res = features['msa_aatype'].shape[:2]
        msa_feat = None
        cluster_msa = features['msa_aatype']
        cluster_has_deletion = (features['msa_deletion_count'] > 0).float().unsqueeze(-1)
        cluster_deletion_value = 2/torch.pi * torch.arctan(features['msa_deletion_count'] / 3)
        cluster_deletion_value = cluster_deletion_value.unsqueeze(-1)
        cluster_deletion_mean = features['cluster_deletion_mean'].unsqueeze(-1)
        cluster_profile = features['cluster_profile']

        msa_feat = torch.cat((cluster_msa, cluster_has_deletion, cluster_deletion_value, cluster_profile, cluster_deletion_mean), dim=-1)
        return msa_feat
    
    def calculate_extra_msa_feat(self, features):
        N_extra, N_res = features['extra_msa_aatype'].shape[:2]
        extra_msa_feat = None

        padding = torch.zeros((N_extra, N_res, 1))
        extra_msa = torch.cat((features['extra_msa_aatype'], padding), dim=-1)
        extra_msa_has_deletion = (features['extra_msa_deletion_count'] > 0 ).float().unsqueeze(-1)
        extra_msa_deletion_value = 2/torch.pi * torch.arctan(features['extra_msa_deletion_count']/3)
        extra_msa_deletion_value = extra_msa_deletion_value.unsqueeze(-1)

        extra_msa_feat = torch.cat((extra_msa, extra_msa_has_deletion, extra_msa_deletion_value), dim=-1)

        return extra_msa_feat

    def create_features_from_a3m(self, file_name, temp_Ph_vals, seed=None):
        msa_feat = None
        extra_msa_feat = None
        target_feat = None
        residue_index = None
        select_clusters_seed = None
        mask_clusters_seed = None
        crop_extra_seed = None
        if seed is not None:
            select_clusters_seed = seed
            mask_clusters_seed = seed+1
            crop_extra_seed = seed+2

        seqs = self.load_a3m_file(file_name)
        features = self.initial_data_from_seqs(seqs)

        transforms = [
            lambda x: self.select_cluster_centers(x, seed=select_clusters_seed),
            lambda x: self.mask_cluster_centers(x, seed=mask_clusters_seed),
            self.cluster_assignment,
            self.summarize_clusters,
            lambda x: self.crop_extra_msa(x, seed=crop_extra_seed)
        ]

        for transform in transforms:
            features = transform(features)

        msa_feat = self.calculate_msa_feat(features)
        extra_msa_feat = self.calculate_extra_msa_feat(features)
        target_feat = self.onehot_encode_aa_type(seqs[0], include_gap_token=False).float()
        residue_index = torch.arange(len(seqs[0]))

        return {
            'msa_feat': msa_feat,
            'extra_msa_feat': extra_msa_feat,
            'target_feat': target_feat,
            'residue_index': residue_index,
            'seq_name' : file_name[14:-4],
            'pH' : temp_Ph_vals[file_name[14:-4]][0],
            'temp': temp_Ph_vals[file_name[14:-4]][1]
            # 'coordinates' :   # Nres, 37, 3 --> target
        }


# t = ProcessDataset()
# print(t.features)
