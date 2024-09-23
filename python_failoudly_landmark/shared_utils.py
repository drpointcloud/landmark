# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from enum import Enum

# -------------------------------------------------
# ENUMS
# -------------------------------------------------

class TestDimensionality(Enum):
    One = 1
    Multi = 2
    Bin = 3


class OnedimensionalTest(Enum):
    KS = 1
    AD = 2
    LMSW = 3 #added


class MultidimensionalTest(Enum):
    MMD = 1
    KNN = 2
    FR = 3
    Energy = 4
    LMSW = 5 #added


class DimensionalityReduction(Enum):
    NoRed = 0
    PCA = 1
    SRP = 2
    UAE = 3
    TAE = 4
    BBSDs = 5
    Inception = 6
    BBSDh = 7
    Classif = 8
    
