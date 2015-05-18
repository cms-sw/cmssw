import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cfi import *

# use the old H/E cut (as in 620_SLHC10, i.e. seed = 3x3 cluster)), 
# since the new one (seed tower) is inefficient at PU140

L1CaloProtoClusterSharing.hoeCutMode = cms.int32(1)    # 0 = seed tower H/E, 1 = 3x3 cluster H/E


SLHCCaloTrigger  = cms.Sequence(#L1TestPatternCaloTowerProducer+
                                L1CaloTowerProducer+
                                #L1RingSubtractionProducer+
                                L1CaloRegionProducer+
                                L1CaloClusterProducer+
                                L1CaloClusterFilter+
                                L1CaloClusterIsolator+
                                L1CaloClusterEGFilter+
                                L1CaloClusterEGIsolator+
                                ## New e/g clustering
                                L1CaloProtoClusterProducer+
                                L1CaloProtoClusterFilter+
                                L1CaloProtoClusterSharing+
                                L1CaloEgammaClusterProducer+
                                L1CaloEgammaClusterIsolator+
                                ## End new e/g clustering
                                L1CaloJetProducer+
                                L1CaloJetFilter+
                                L1TowerJetProducer+
                                L1TowerJetCentralityFilter+
                                #L1TowerJetFilter1D+
                                #L1TowerJetFilter2D+
                                L1TowerJetPUEstimator+
                                L1TowerJetPUSubtractedProducer+
                                L1CalibFilterTowerJetProducer+
                                L1EnergySumProducer+ 

                                # L1TowerFwdJetProducer+
                                # L1TowerFwdJetFilter1D+
                                # L1TowerFwdJetFilter2D+
                                L1CaloJetExpander+
                                rawSLHCL1ExtraParticles+
                                rawSLHCL1ExtraParticlesNewClustering+
                                SLHCL1ExtraParticles+
                                SLHCL1ExtraParticlesNewClustering
                             
                                #l1extraParticlesCalibrated
)

#uncomment the lines below for verbose (Huge amount of printouts!)
#L1CaloClusterProducer.verbosity = cms.untracked.bool(True)
#L1CaloClusterFilter.verbosity = cms.untracked.bool(True)
#L1CaloJetProducer.verbosity = cms.untracked.bool(True)
#L1CaloJetFilter.verbosity = cms.untracked.bool(True)
#L1CaloJetExpander.verbosity = cms.untracked.bool(True)



