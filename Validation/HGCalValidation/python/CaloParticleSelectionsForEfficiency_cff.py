import FWCore.ParameterSet.Config as cms

generalCpSelectorBlock = cms.PSet(
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    intimeOnly = cms.bool(False),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-4.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    ptMax = cms.double(1e100),
    maxRapidity = cms.double(4.5),
    tip = cms.double(3.5),
    minPhi = cms.double(-3.2),
    maxPhi = cms.double(3.2),
)

CpSelectorForEfficiencyVsEtaBlock = generalCpSelectorBlock.clone()
CpSelectorForEfficiencyVsPhiBlock = generalCpSelectorBlock.clone()
CpSelectorForEfficiencyVsPtBlock = generalCpSelectorBlock.clone(ptMin = 0.050 )

