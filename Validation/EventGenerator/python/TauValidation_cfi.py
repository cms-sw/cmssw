import FWCore.ParameterSet.Config as cms

tauValidation = DQMStep1Module('TauValidation',
#                               hepmcCollection = cms.InputTag("genParticles",""),
                               genparticleCollection = cms.InputTag("genParticles",""),
                               tauEtCutForRtau = cms.double(50),
                               UseWeightFromHepMC = cms.bool(True)
                               )
