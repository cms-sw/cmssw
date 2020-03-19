import FWCore.ParameterSet.Config as cms
from Validation.RecoParticleFlow.PFCluster_cfi import pfclusterAnalyzer
 
pfClusterValidationSequence = cms.Sequence( pfclusterAnalyzer )
