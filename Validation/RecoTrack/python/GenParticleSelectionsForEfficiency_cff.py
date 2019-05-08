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
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase1Pixel.toModify(GenParticleSelectionForEfficiency,minRapidityGP = -3.0, maxRapidityGP = 3.0)
phase2_tracker.toModify(GenParticleSelectionForEfficiency,minRapidityGP = -4.5, maxRapidityGP = 4.5)

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

def _modifyForPhase1(pset):
    pset.minRapidity = -3
    pset.maxRapidity = 3
    pset.tip = 2.5 # beampipe is around 2.0, BPIX1 is at 2.9
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(generalGpSelectorBlock,             _modifyForPhase1)
phase1Pixel.toModify(GpSelectorForEfficiencyVsEtaBlock,  _modifyForPhase1)
phase1Pixel.toModify(GpSelectorForEfficiencyVsPhiBlock,  _modifyForPhase1)
phase1Pixel.toModify(GpSelectorForEfficiencyVsPtBlock,   _modifyForPhase1)
phase1Pixel.toModify(GpSelectorForEfficiencyVsVTXRBlock, _modifyForPhase1)
phase1Pixel.toModify(GpSelectorForEfficiencyVsVTXZBlock, _modifyForPhase1)

def _modifyForPhase2(pset):
    pset.minRapidity = -4.5
    pset.maxRapidity = 4.5
    pset.tip = 2.5 # IT1 will be around 3.0 (as in Phase1)
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(generalGpSelectorBlock,             _modifyForPhase2)
phase2_tracker.toModify(GpSelectorForEfficiencyVsEtaBlock,  _modifyForPhase2)
phase2_tracker.toModify(GpSelectorForEfficiencyVsPhiBlock,  _modifyForPhase2)
phase2_tracker.toModify(GpSelectorForEfficiencyVsPtBlock,   _modifyForPhase2)
phase2_tracker.toModify(GpSelectorForEfficiencyVsVTXRBlock, _modifyForPhase2)
phase2_tracker.toModify(GpSelectorForEfficiencyVsVTXZBlock, _modifyForPhase2)
