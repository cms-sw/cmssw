import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
source = cms.Source(
    "PythiaSource",
    pythiaVerbosity = cms.untracked.bool(False),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(1),
    DoubleParticle = cms.untracked.bool(True),
    Ptmin = cms.untracked.double(20.0),
    Ptmax = cms.untracked.double(700.0),
#    Emin = cms.untracked.double(10.0),
#    Emax = cms.untracked.double(10.0),
    Etamin = cms.untracked.double(0.0),
    Etamax = cms.untracked.double(1.0),
    Phimin = cms.untracked.double(0.0),
    Phimax = cms.untracked.double(360.0),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        # Tau jets only
        pythiaJets = cms.vstring(),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring(
            'pythiaUESettings',
            'pythiaJets'
        )

    )
    
)




