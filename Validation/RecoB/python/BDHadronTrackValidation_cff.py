import FWCore.ParameterSet.Config as cms

# Need to add pfImpactParameterTagInfos to the pat jets --> make personal patJet collection
# and rerun PAT (because sometimes validation is ran without PAT sequence enables)
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
patJetsBDHadron = patJets.clone(
    tagInfoSources = cms.VInputTag(cms.InputTag('pfImpactParameterTagInfos')),
    addTagInfos = cms.bool(True)
)

# my analyzer
from Validation.RecoB.BDHadronTrackMonitoring_cfi import *
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
BDHadronTrackMonitoringAnalyze.PatJetSource = cms.InputTag('patJetsBDHadron')

bdHadronTrackValidationSeq = cms.Sequence(BDHadronTrackMonitoringAnalyze,
                                          cms.Task(patJetCorrectionsTask
                                                   , patJetCharge
                                                   , patJetPartonMatch
                                                   , patJetGenJetMatch
                                                   , patJetFlavourIdLegacyTask
                                                   , patJetFlavourIdTask
                                                   , patJetsBDHadron
                                                   , tpClusterProducer)
) 

					
bdHadronTrackPostProcessor = cms.Sequence(BDHadronTrackMonitoringHarvest)
