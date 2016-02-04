import FWCore.ParameterSet.Config as cms

import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
cutsTPFake = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
cutsTPFake.ptMin = 0.1
cutsTPFake.minRapidity = -3.0
cutsTPFake.maxRapidity = 3.0
cutsTPFake.tip = 120
cutsTPFake.lip = 250
cutsTPFake.signalOnly = False


