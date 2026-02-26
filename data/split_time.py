import numpy as np

def time_based_split(timestep):
    """
    Elliptic dataset time-based split.
    
    Train: timestep <= 34
    Val:   35–40
    Test:  > 40
    """

    train_mask = timestep <= 34
    val_mask   = (timestep > 34) & (timestep <= 40)
    test_mask  = timestep > 40

    return train_mask, val_mask, test_mask
