import FWCore.ParameterSet.Config as cms
import SimTracker.TrackAssociation.tpSelectorByGenDefault_cfi as _mod

prunedTrackingParticles = _mod.tpSelectorByGenDefault.clone(
    select = [
        "drop  *", # this is the default
        "keep++ (400 < abs(pdgId) < 600) || (4000 < abs(pdgId) < 6000)",   # keep decays for BPH studies
        "drop status != 1",                                                # keep only status == 1
        "drop charge == 0"
    ]
)
