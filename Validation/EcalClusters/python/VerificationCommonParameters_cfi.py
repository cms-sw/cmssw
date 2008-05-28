import FWCore.ParameterSet.Config as cms

VerificationCommonParameters = cms.PSet(
    CMSSW_Version = cms.untracked.string('V2_0_0_pre6'),
    MCTruthCollection = cms.InputTag("source"),
    outputFile = cms.untracked.string('EcalClustersValidation.root'),
    verboseDBE = cms.untracked.bool(True)
)


