import FWCore.ParameterSet.Config as cms

import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
myTrackingParticle = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
myTrackingParticle.chargedOnly = True
# trackingParticleSelector.pdgId = cms.vint32()
myTrackingParticle.tip = 120
myTrackingParticle.lip = 280
myTrackingParticle.signalOnly = False
myTrackingParticle.minRapidity = -2.5
myTrackingParticle.ptMin =0.9
myTrackingParticle.maxRapidity = 2.5
myTrackingParticle.minHit = 0


