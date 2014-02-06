import FWCore.ParameterSet.Config as cms

generalTpSelectorBlock = cms.PSet(
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-4.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(4.5),
    tip = cms.double(3.5)
)


TpSelectorForEfficiencyVsEtaBlock = cms.PSet(
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-4.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(4.5),
    tip = cms.double(3.5)
)

TpSelectorForEfficiencyVsPhiBlock = cms.PSet(
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-2.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    tip = cms.double(3.5)
)

TpSelectorForEfficiencyVsPtBlock = cms.PSet(
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-2.5),
    maxRapidity = cms.double(2.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.050),
    tip = cms.double(3.5),
    lip = cms.double(30.0),
)

TpSelectorForEfficiencyVsVTXRBlock = cms.PSet(
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-2.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    lip = cms.double(30.0),
    tip = cms.double(60.0)
)

TpSelectorForEfficiencyVsVTXZBlock = cms.PSet(
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-2.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    lip = cms.double(30.0),
    tip = cms.double(3.5)
)
