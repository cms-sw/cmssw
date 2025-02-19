import FWCore.ParameterSet.Config as cms

import PhysicsTools.RecoAlgos.trackingParticleSelector_cfi
tpSelection = PhysicsTools.RecoAlgos.trackingParticleSelector_cfi.trackingParticleSelector.clone()
tpSelection.chargedOnly = True
# trackingParticleSelector.pdgId = cms.vint32()
tpSelection.tip = 120
tpSelection.lip = 280
tpSelection.signalOnly = False
tpSelection.minRapidity = -2.5
tpSelection.ptMin = 1.
tpSelection.maxRapidity = 2.5
tpSelection.minHit = 0


