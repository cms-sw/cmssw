import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
source = cms.Source("PythiaSource",
    Phimin = cms.untracked.double(0.0),
    #  possibility to run single or double back-to-back particles with PYTHIA
    # if ParticleID = 0, run PYTHIA
    ParticleID = cms.untracked.int32(15),
    Etamin = cms.untracked.double(-1.0),
    DoubleParticle = cms.untracked.bool(False),
    Phimax = cms.untracked.double(360.0),
    Ptmin = cms.untracked.double(200.0),
    Ptmax = cms.untracked.double(500.0001),
    Etamax = cms.untracked.double(1.0),
    pythiaVerbosity = cms.untracked.bool(False),
    PythiaParameters = cms.PSet(
        pythiaUESettingsBlock,
        pythiaTauJets = cms.vstring('MDME(89,1)=0      ! no tau->electron', 
            'MDME(90,1)=0      ! no tau->muon'),
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring('pythiaUESettings', 
            'pythiaTauJets')
    )
)



