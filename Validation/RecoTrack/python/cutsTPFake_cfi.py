import FWCore.ParameterSet.Config as cms

from CommonTools.RecoAlgos.trackingParticleRefSelector_cfi import trackingParticleRefSelector as _trackingParticleRefSelector
cutsTPFake = _trackingParticleRefSelector.clone(
    ptMin = 0.1,
    minRapidity = -3.0,
    maxRapidity = 3.0,
    tip = 120,
    lip = 250,
    signalOnly = False
)

