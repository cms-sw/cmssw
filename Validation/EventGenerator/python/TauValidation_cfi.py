import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
tauValidation = DQMEDAnalyzer('TauValidation',
#                               hepmcCollection = cms.InputTag("genParticles",""),
                               genparticleCollection = cms.InputTag("genParticles",""),
                               tauEtCutForRtau = cms.double(50),
                               UseWeightFromHepMC = cms.bool(True)
                               )
