import FWCore.ParameterSet.Config as cms

# Need to add pfImpactParameterTagInfos to the pat jets --> make personal patJet collection
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cfi import _patJets as patJets
patJetsBDHadron = patJets.clone(
    tagInfoSources = cms.VInputTag(cms.InputTag('pfImpactParameterTagInfos')),
    addTagInfos = cms.bool(True)
)

selectedPatJetsBDHadron = cms.EDFilter("PATJetSelector",
    src = cms.InputTag("patJetsBDHadron"),
    cut = cms.string("pt > 10.")
)


# my analyzer
from Validation.RecoB.BDHadronTrackMonitoring_cfi import *
BDHadronTrackMonitoringAnalyze.PatJetSource = cms.InputTag('selectedPatJetsBDHadron')#'selectedPatJets')



bdHadronTrackValidationSeq = cms.Sequence(patJetsBDHadron * selectedPatJetsBDHadron * BDHadronTrackMonitoringAnalyze)

bdHadronTrackPostProcessor = cms.Sequence(BDHadronTrackMonitoringHarvest)
