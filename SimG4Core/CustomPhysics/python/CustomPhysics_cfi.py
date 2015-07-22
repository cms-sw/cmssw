import FWCore.ParameterSet.Config as cms

customPhysicsSetup = cms.PSet(
    reggeSuppression = cms.double(0.0), ##

    # Paths to files with particle and physics processes definition
    particlesDef = cms.FileInPath('SimG4Core/CustomPhysics/data/particles_gluino_300_GeV.txt'),
    resonanceEnergy = cms.double(200.0), ##

    #        FileInPath particlesDef = "SimG4Core/CustomPhysics/data/isa-slha.out"
    processesDef = cms.FileInPath('SimG4Core/CustomPhysics/data/RhadronProcessList.txt'),
    amplitude = cms.double(100.0), ##

    # R-hadron physics setup
    rhadronPhysics = cms.bool(True),
    resonant = cms.bool(False),
    gamma = cms.double(0.1),
    reggeModel = cms.bool(False),
    hadronLifeTime = cms.double(-1.),
    mixing = cms.double(1.)

)
