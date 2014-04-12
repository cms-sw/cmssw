import FWCore.ParameterSet.Config as cms

GenParticleSelectionForEfficiency = cms.PSet(
    lipGP = cms.double(30.0),
    chargedOnlyGP = cms.bool(True),
    pdgIdGP = cms.vint32(),
    minRapidityGP = cms.double(-2.5),
    ptMinGP = cms.double(0.005),
    maxRapidityGP = cms.double(2.5),
    tipGP = cms.double(60),
    statusGP = cms.int32(1)
)

generalGpSelectorBlock = cms.PSet(
    status = cms.int32(1),
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    minRapidity = cms.double(-2.5),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    tip = cms.double(3.5)
)


GpSelectorForEfficiencyVsEtaBlock = cms.PSet(
    status = cms.int32(1),
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    minRapidity = cms.double(-2.5),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    tip = cms.double(3.5)
)

GpSelectorForEfficiencyVsPhiBlock = cms.PSet(
    status = cms.int32(1),
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    minRapidity = cms.double(-2.5),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    tip = cms.double(3.5)
)

GpSelectorForEfficiencyVsPtBlock = cms.PSet(
    status = cms.int32(1),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    minRapidity = cms.double(-2.5),
    maxRapidity = cms.double(2.5),
    ptMin = cms.double(0.050),
    tip = cms.double(3.5),
    lip = cms.double(30.0),
)

GpSelectorForEfficiencyVsVTXRBlock = cms.PSet(
    status = cms.int32(1),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    minRapidity = cms.double(-2.5),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    lip = cms.double(30.0),
    tip = cms.double(30.0)
)

GpSelectorForEfficiencyVsVTXZBlock = cms.PSet(
    status = cms.int32(1),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    minRapidity = cms.double(-2.5),
    ptMin = cms.double(0.9),
    maxRapidity = cms.double(2.5),
    lip = cms.double(35.0),
    tip = cms.double(3.5)
)
