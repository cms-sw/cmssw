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
    subLeadLepPtCut = cms.double(10.),
    lepEtaCut = cms.double(2.4),
    FSRdRCut = cms.double(0.1),
    ZptRange = cms.double(200.),
    ZptNbin = cms.int32(100),
    ZmassLow = cms.double(60.),
    ZmassHigh = cms.double(120.),
    ZmassNbin = cms.int32(120),
    rapidityRange = cms.double(2.4),
    rapidityNbin = cms.int32(120),
    jetPtCut = cms.double(20.),
    jetEtaCut = cms.double(2.4),
    nJetsNbin = cms.int32(7),
    jetPtRange = cms.double(200.),
    jetPtNbin = cms.int32(100)
)

genWeightValidationSeq = cms.Sequence(genWeightValidation)
