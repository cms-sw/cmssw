import FWCore.ParameterSet.Config as cms

# First register all the clustering algorithms, then specify preferred
# ones at end.

# Clustering algorithm a
ClusteringAlgorithm_a_PSimHit_   = cms.ESProducer("ClusteringAlgorithm_a_PSimHit_")
ClusteringAlgorithm_a_PixelDigi_ = cms.ESProducer("ClusteringAlgorithm_a_PixelDigi_")

# Broadside clustering algorithm
# Set WidthCut=0 to eliminate the width cut.
#ClusteringAlgorithm_broadside_PSimHit_ = cms.ESProducer("ClusteringAlgorithm_broadside_PSimHit_",
#    WidthCut = cms.int32(3)
#)  
ClusteringAlgorithm_broadside_PixelDigi_ = cms.ESProducer("ClusteringAlgorithm_broadside_PixelDigi_",
    WidthCut = cms.int32(4)
)

# 2d clustering algorithm
#ClusteringAlgorithm_2d_PSimHit_ = cms.ESProducer("ClusteringAlgorithm_2d_PSimHit_",
#    DoubleCountingTest=cms.bool(False)
#)
ClusteringAlgorithm_2d_PixelDigi_ = cms.ESProducer("ClusteringAlgorithm_2d_PixelDigi_",
    DoubleCountingTest=cms.bool(True)
)

# Neighbor clustering algorithm
ClusteringAlgorithm_neighbor_PixelDigi_ = cms.ESProducer("ClusteringAlgorithm_neighbor_PixelDigi_")

# Set the preferred hit matching algorithms.
# We prefer the a algorithm for now in order not to break anything.
# Override with process.ClusteringAlgorithm_PSimHit_ = ..., etc. in your
# configuration.
ClusteringAlgorithm_PSimHit_   = cms.ESPrefer("ClusteringAlgorithm_a_PSimHit_")
ClusteringAlgorithm_PixelDigi_ = cms.ESPrefer("ClusteringAlgorithm_broadside_PixelDigi_")

