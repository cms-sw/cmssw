import FWCore.ParameterSet.Config as cms

def customise(process):

    FLAVOR = process.generator.hscpFlavor.value()
    PROCESS_FILE = process.generator.processFile.value()
    PARTICLE_FILE = process.generator.particleFile.value()
    USE_REGGE = process.generator.useregge.value()

    process.load("SimG4Core.CustomPhysics.CustomPhysics_cfi")
    process.customPhysicsSetup.particlesDef = PARTICLE_FILE
    process.customPhysicsSetup.reggeModel = USE_REGGE

    # Read in the SLHA file to get the particles definition
    if hasattr(process, 'generator') and hasattr(process.generator, 'SLHAFileForPythia8'):
        process.customPhysicsSetup.particlesDef = process.generator.SLHAFileForPythia8.value()
    elif hasattr(process, 'g4SimHits') and hasattr(process.g4SimHits, 'SLHAFileForPythia8'):
        process.customPhysicsSetup.particlesDef = process.g4SimHits.SLHAFileForPythia8.value()

    # Passing pythia settings to the Rhadron decayer is optional
    if hasattr(process, 'generator') and hasattr(process.generator, 'RhadronPythiaDecayerCommandFile'):
        process.customPhysicsSetup.RhadronPythiaDecayerCommandFile = process.generator.RhadronPythiaDecayerCommandFile.value()
    elif hasattr(process, 'g4SimHits') and hasattr(process.g4SimHits, 'RhadronPythiaDecayerCommandFile'):
        process.customPhysicsSetup.RhadronPythiaDecayerCommandFile = process.g4SimHits.RhadronPythiaDecayerCommandFile.value()

    if hasattr(process,'g4SimHits'):
        # defined watches
        process.g4SimHits.Watchers = cms.VPSet (
            cms.PSet(
                type = cms.string('RHStopTracer'),
                RHStopTracer = cms.PSet(
                verbose = cms.untracked.bool (False),
                traceParticle = cms.string ("((anti_)?~|tau1).*"), #this one regular expression is needed to look for ~HIP*, anti_~HIP*, ~tau1, anti_~tau1, ~g_rho0, ~g_Deltabar0, ~T_uu1++, etc
                stopRegularParticles = cms.untracked.bool (False)
                )        
            )
        )
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

        # Add Rhadron decay tracking
        process.RHDecayTracer = cms.EDProducer("RHDecayTracer",
            RHDecayTracerPDGs = cms.untracked.vstring("1000600-1999999", "1000021", "1000006")
        )
        process.simulation_step *= process.RHDecayTracer

        return (process)