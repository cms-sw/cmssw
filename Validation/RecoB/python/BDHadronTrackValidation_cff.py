import FWCore.ParameterSet.Config as cms

# Need to add pfImpactParameterTagInfos to the pat jets --> make personal patJet collection
# and rerun PAT (because sometimes validation is ran without PAT sequence enables)
from PhysicsTools.PatAlgos.producersLayer1.jetProducer_cff import *
patJetsBDHadron = patJets.clone(
    tagInfoSources = cms.VInputTag(cms.InputTag('pfImpactParameterTagInfos')),
    addTagInfos = cms.bool(True)
)

#selectedPatJetsBDHadron = cms.EDFilter("PATJetSelector",
#    src = cms.InputTag("patJetsBDHadron"),
#    cut = cms.string("pt > 10.")
#)


# my analyzer
from Validation.RecoB.BDHadronTrackMonitoring_cfi import *
from SimTracker.TrackerHitAssociation.tpClusterProducer_cfi import *
BDHadronTrackMonitoringAnalyze.PatJetSource = cms.InputTag('patJetsBDHadron')#'selectedPatJetsBDHadron')



bdHadronTrackValidationSeq = cms.Sequence(patJetCorrections
					* patJetCharge*patJetPartonMatch
					* patJetGenJetMatch*patJetFlavourIdLegacy
					* patJetFlavourId*patJetsBDHadron
					* tpClusterProducer
					* BDHadronTrackMonitoringAnalyze) #patJetsBDHadron * selectedPatJetsBDHadron * 

bdHadronTrackPostProcessor = cms.Sequence(BDHadronTrackMonitoringHarvest)
