import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
lheWeightValidation = DQMEDAnalyzer('LheWeightValidation',
    lheProduct = cms.InputTag('externalLHEProducer'),
    genParticles = cms.InputTag('genParticles'),
    genJets = cms.InputTag('ak4GenJets'),
    dumpLHEheader = cms.bool(False),
    nScaleVar = cms.int32(9),
    idxPdfStart = cms.int32(9),
    idxPdfEnd = cms.int32(111),
    leadLepPtRange = cms.double(200.),
    leadLepPtNbin = cms.int32(100),
    leadLepPtCut = cms.double(20.),
    lepEtaCut = cms.double(2.4),
    rapidityRange = cms.double(2.4),
    rapidityNbin = cms.int32(120),
    jetPtCut = cms.double(20.),
    jetEtaCut = cms.double(2.4),
    nJetsNbin = cms.int32(20),
    jetPtRange = cms.double(200.),
    jetPtNbin = cms.int32(100)
)

lheWeightValidationSeq = cms.Sequence(lheWeightValidation)
