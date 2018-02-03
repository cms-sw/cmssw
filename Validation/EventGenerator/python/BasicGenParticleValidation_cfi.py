import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
basicGenParticleValidation = DQMEDAnalyzer('BasicGenParticleValidation',
    hepmcCollection = cms.InputTag("generatorSmeared"),
    genparticleCollection = cms.InputTag("genParticles",""),
    genjetsCollection = cms.InputTag("ak4GenJets",""),
    matchingPrecision = cms.double(0.001),
    verbosity = cms.untracked.uint32(0),
    UseWeightFromHepMC = cms.bool(True)
)
