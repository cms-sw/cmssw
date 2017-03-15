import FWCore.ParameterSet.Config as cms

# my analyzer
from Validation.RecoB.BDHadronTrackMonitoring_cfi import *
BDHadronTrackMonitoringAnalyze.PatJetSource = cms.InputTag('selectedPatJets')

bdHadronTrackValidationSeq = cms.Sequence(BDHadronTrackMonitoringAnalyze)

bdHadronTrackPostProcessor = cms.Sequence(BDHadronTrackMonitoringHarvest)
