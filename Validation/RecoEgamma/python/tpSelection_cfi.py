import FWCore.ParameterSet.Config as cms

import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
tpSelection = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone(
    chargedOnly = True,
    # trackingParticleSelector.pdgId = cms.vint32()
    tip = 120,
    lip = 280,
    signalOnly = False,
    minRapidity = -2.5,
    ptMin = 1.0,
    maxRapidity = 2.5,
    minHit = 0

)

