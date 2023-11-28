import FWCore.ParameterSet.Config as cms

TrackingParticleSelectionForEfficiency = cms.PSet(
    lipTP = cms.double(30.0),
    chargedOnlyTP = cms.bool(True),
    stableOnlyTP = cms.bool(False),
    pdgIdTP = cms.vint32(),
    signalOnlyTP = cms.bool(False),
    intimeOnlyTP = cms.bool(True),
    minRapidityTP = cms.double(-2.5),
    minHitTP = cms.int32(0),
    ptMinTP = cms.double(0.005),
    ptMaxTP = cms.double(1e100),
    maxRapidityTP = cms.double(2.5),
    tipTP = cms.double(60),
    invertRapidityCutTP = cms.bool(False),
    maxPhi = cms.double(3.2),
    minPhi = cms.double(-3.2),
    applyTPSelToSimMatch = cms.bool(False)
)

def _modifyForPhase1(pset):
    pset.minRapidityTP = -3
    pset.maxRapidityTP = 3
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
phase1Pixel.toModify(TrackingParticleSelectionForEfficiency, _modifyForPhase1)
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase2_tracker.toModify(TrackingParticleSelectionForEfficiency, minRapidityTP = -4.5, maxRapidityTP = 4.5)
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toModify(TrackingParticleSelectionForEfficiency, stableOnlyTP = True)
    
