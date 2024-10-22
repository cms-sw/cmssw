import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
genWeightValidation = DQMEDAnalyzer('GenWeightValidation',
    UseWeightFromHepMC = cms.bool(False),
    genEventInfos = cms.VInputTag('generator'),
    genParticles = cms.InputTag('genParticles'),
    genJets = cms.InputTag('ak4GenJets'),
    whichGenEventInfo = cms.int32(0),
    idxFSRup = cms.int32(5),
    idxFSRdown = cms.int32(4),
    idxISRup = cms.int32(27),
    idxISRdown = cms.int32(26),
    leadLepPtRange = cms.double(200.),
    leadLepPtNbin = cms.int32(100),
    leadLepPtCut = cms.double(20.),
    lepEtaCut = cms.double(2.4),
    jetPtCut = cms.double(20.),
    rapidityRange = cms.double(2.4),
    rapidityNbin = cms.int32(120),
    jetEtaCut = cms.double(2.4),
    nJetsNbin = cms.int32(20),
    jetPtRange = cms.double(200.),
    jetPtNbin = cms.int32(100)
)

genWeightValidationSeq = cms.Sequence(genWeightValidation)
