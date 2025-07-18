import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker

generalTpSelectorBlock = cms.PSet(
    lip = cms.double(30.0),
    chargedOnly = cms.bool(True),
    pdgId = cms.vint32(),
    signalOnly = cms.bool(True),
    intimeOnly = cms.bool(False),
    stableOnly = cms.bool(False),
    minRapidity = cms.double(-2.5),
    minHit = cms.int32(0),
    ptMin = cms.double(0.9),
    ptMax = cms.double(1e100),
    maxRapidity = cms.double(2.5),
    tip = cms.double(3.5),
    minPhi = cms.double(-3.2),
    maxPhi = cms.double(3.2),
    invertRapidityCut = cms.bool(False)
)

from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(generalTpSelectorBlock, stableOnly = True)

generalTpSelectorForEfficiencyBlock = generalTpSelectorBlock.clone() # TP selector block for efficiency (with additional selections)

def _modifyForPhase1Efficiency(pset):
    pset.tip = 2.5 # beampipe is around 2.0, BPIX1 is at 2.9

phase1Pixel.toModify(generalTpSelectorForEfficiencyBlock, _modifyForPhase1Efficiency)

def _modifyForPhase2Efficiency(pset):
    pset.minRapidity = -3.5 # within efficient eta range in phase-2
    pset.maxRapidity = 3.5 # within efficient eta range in phase-2
    pset.tip = 2.5 # IT1 will be around 3.0 (as in Phase1)

phase2_tracker.toModify(generalTpSelectorForEfficiencyBlock, _modifyForPhase2Efficiency)

TpSelectorForEfficiencyVsEtaBlock = generalTpSelectorForEfficiencyBlock.clone()
TpSelectorForEfficiencyVsPhiBlock = generalTpSelectorForEfficiencyBlock.clone()
TpSelectorForEfficiencyVsPtBlock = generalTpSelectorForEfficiencyBlock.clone(ptMin = 0.050 )
TpSelectorForEfficiencyVsVTXRBlock = generalTpSelectorForEfficiencyBlock.clone(tip = 60.0)
TpSelectorForEfficiencyVsVTXZBlock = generalTpSelectorForEfficiencyBlock.clone()

def _modifyForPhase1(pset):
    pset.minRapidity = -3
    pset.maxRapidity = 3
    pset.tip = 2.5 # beampipe is around 2.0, BPIX1 is at 2.9

phase1Pixel.toModify(generalTpSelectorBlock, _modifyForPhase1) # for general TP selector, extend eta to full acceptance
phase1Pixel.toModify(TpSelectorForEfficiencyVsEtaBlock, _modifyForPhase1) # for efficiency vs eta, also extend eta to full acceptance

def _modifyForPhase2(pset):
    pset.minRapidity = -4.5
    pset.maxRapidity = 4.5
    pset.tip = 2.5 # IT1 will be around 3.0 (as in Phase1)

phase2_tracker.toModify(generalTpSelectorBlock, _modifyForPhase2) # for general TP selector, extend eta to full acceptance
phase2_tracker.toModify(TpSelectorForEfficiencyVsEtaBlock, _modifyForPhase2) # for efficiency vs eta, also extend eta to full acceptance
