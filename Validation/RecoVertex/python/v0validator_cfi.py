import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
v0Validator = DQMEDAnalyzer('V0Validator',
    DQMRootFileName = cms.untracked.string(''),
    kShortCollection = cms.untracked.InputTag('generalV0Candidates:Kshort'),
    lambdaCollection = cms.untracked.InputTag('generalV0Candidates:Lambda'),
    trackAssociatorMap = cms.untracked.InputTag("trackingParticleRecoTrackAsssociation"),
    trackingVertexCollection = cms.untracked.InputTag("mix", "MergedTrackTruth"),
    vertexCollection = cms.untracked.InputTag("offlinePrimaryVertices"),
    dirName = cms.untracked.string('Vertexing/V0V')
)

from Configuration.ProcessModifiers.premix_stage2_cff import premix_stage2
premix_stage2.toModify(v0Validator, trackingVertexCollection = "mixData:MergedTrackTruth")
