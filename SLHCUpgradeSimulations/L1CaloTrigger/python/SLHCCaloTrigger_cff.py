import FWCore.ParameterSet.Config as cms

from SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cfi import *

SLHCCaloTrigger  = cms.Sequence(L1CaloTowerProducer+
							#	L1RingSubtractionProducer+
                                L1CaloRegionProducer+
                                L1CaloClusterProducer+
                                L1CaloClusterFilter+
                                L1CaloClusterIsolator+
                                L1CaloJetProducer+
                                L1CaloJetFilter+
                                L1CaloJetExpander+
								L1TowerJetProducer+
                                rawSLHCL1ExtraParticles+
                                SLHCL1ExtraParticles+
                                l1extraParticlesCalibrated
)

#uncomment the lines below for verbose (Huge amount of printouts!)
#L1CaloClusterProducer.verbosity = cms.untracked.bool(True)
#L1CaloClusterFilter.verbosity = cms.untracked.bool(True)
#L1CaloJetProducer.verbosity = cms.untracked.bool(True)
#L1CaloJetFilter.verbosity = cms.untracked.bool(True)
#L1CaloJetExpander.verbosity = cms.untracked.bool(True)



