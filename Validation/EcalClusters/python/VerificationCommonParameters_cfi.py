import FWCore.ParameterSet.Config as cms

VerificationCommonParameters = cms.PSet(
#    CMSSW_Version = cms.untracked.string('V3_1_0_pre1'),
    MCTruthCollection = cms.InputTag("generator"),
#    outputFile = cms.untracked.string('EcalClustersValidation.root'),
    verboseDBE = cms.untracked.bool(False)
)


