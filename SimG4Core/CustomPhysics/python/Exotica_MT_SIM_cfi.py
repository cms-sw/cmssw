from __future__ import print_function
import FWCore.ParameterSet.Config as cms

def customise(process):

    FLAVOR = process.generator.hscpFlavor.value()
    PROCESS_FILE = process.generator.processFile.value()
    PARTICLE_FILE = process.generator.particleFile.value()
    USE_REGGE = process.generator.useregge.value()

    process.load("SimG4Core.CustomPhysics.CustomPhysics_cfi")
    process.customPhysicsSetup.particlesDef = PARTICLE_FILE
    process.customPhysicsSetup.reggeModel = USE_REGGE

    if hasattr(process,'g4SimHits'):
        # defined custom Physics List
        process.g4SimHits.Physics.type = cms.string('SimG4Core/Physics/CustomPhysics')
        # add verbosity
        process.g4SimHits.Physics.Verbosity = cms.untracked.int32(0)
        #process.g4SimHits.G4Commands = cms.vstring("/control/cout/ignoreThreadsExcept 0")
        # check flavor of exotics and choose exotica Physics List
        if FLAVOR=="gluino" or FLAVOR=="stop":
            process.customPhysicsSetup.processesDef = PROCESS_FILE
            process.g4SimHits.Physics.ExoticaPhysicsSS = cms.untracked.bool(False)
        elif FLAVOR =="stau":
            process.g4SimHits.Physics.ExoticaPhysicsSS = cms.untracked.bool(False)
        else:
            print("Wrong flavor %s. Only accepted are gluino, stau, stop." % FLAVOR)
        # add custom options
        process.g4SimHits.Physics = cms.PSet(
            process.g4SimHits.Physics, #keep all default value and add others
            process.customPhysicsSetup
        )	

        return (process)

