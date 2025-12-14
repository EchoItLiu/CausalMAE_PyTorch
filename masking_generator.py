
import random
import math
import numpy as np
import torch




class double_phase_MaskingGenerator1d:

    '''
    Mask Generator - Implementing a Two-Stage Masking Strategy
    '''
    def __init__(self, seq_length=101):
        self.seq_length = seq_length

    def generate_mask(self, sample_size, mask_ratio, phase):
        
        if phase == 1:
            # Stage 1: Random masking, default 0.5, strictly following the Kaiming notation
            # Calculate the number of tokens to retain
            len_keep = int(self.seq_length * (1 - mask_ratio))

            # Generate random noise values ranging from 0 to 1 (each sequence is independent)
            noise = torch.rand(sample_size, self.seq_length)
            
            #Sort the noise and obtain the retention indices
            # The method torch.argsort(noise, dim=1) will sort along the first dimension of the noise tensor (i.e., the row direction) and return the index values of each element in the original tensor at its corresponding position after sorting,
            # Retention: From the shuffled indices, only the first len_keep indices of each row are retained in length
            ids_shuffle = torch.argsort(noise, dim=1)
            ids_keep = ids_shuffle[:, :len_keep]

            ## Create binary mask (0 = keep, 1 = mask)
            # Each element is a boolean value. True == 1
            mask = torch.ones(sample_size, self.seq_length, dtype=torch.bool)
            # Perform the operation in the 1st dimension (column direction), retaining the ids_keep number of samples,
            # and setting 0 to False # B×L
            mask.scatter_(dim=1, index= ids_keep, value=0)

        else:
            # Phase 2: Temporal Causal Masking - Keep only 15%, and mask the remaining 85%
            keep_ratio = 1 - mask_ratio
            keep_length = int(self.seq_length * keep_ratio)
            mask = torch.ones(sample_size, self.seq_length, dtype=torch.bool)
            # Keep the first xx% and mask the entire consecutive sequence that follows
            mask[:, :keep_length] = False
            
        return mask
            







