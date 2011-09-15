import FWCore.ParameterSet.Config as cms

def customise(process):
    
	FLAVOR = process.generator.hscpFlavor.value()
	MASS_POINT = process.generator.massPoint.value()
	SLHA_FILE = process.generator.slhaFile.value()
	PROCESS_FILE = process.generator.processFile.value()
	PARTICLE_FILE = process.generator.particleFile.value()
	

	process.load("SimG4Core.CustomPhysics.CustomPhysics_cfi")
	process.customPhysicsSetup.particlesDef = PARTICLE_FILE

        process.g4SimHits.Watchers = cms.VPSet (
            cms.PSet(
            type = cms.string('RHStopTracer'),
            RHStopTracer = cms.PSet(
            verbose = cms.untracked.bool (False),
            traceParticle = cms.string ("(~|tau1).*"),
            stopRegularParticles = cms.untracked.bool (False)
            )        
            )
            )

	
	if FLAVOR=="gluino" or FLAVOR=="stop":
		process.customPhysicsSetup.processesDef = PROCESS_FILE
		process.g4SimHits.Physics = cms.PSet(
        process.customPhysicsSetup,
        DummyEMPhysics = cms.bool(True),
        G4BremsstrahlungThreshold = cms.double(0.5), ## cut in GeV    
        DefaultCutValue = cms.double(1.), ## cuts in cm,default 1cm    
        CutsPerRegion = cms.bool(True),
        Verbosity = cms.untracked.int32(0),
        type = cms.string('SimG4Core/Physics/CustomPhysics'),
        EMPhysics   = cms.untracked.bool(True),  ##G4 default true
        HadPhysics  = cms.untracked.bool(True),  ##G4 default true
        FlagBERT    = cms.untracked.bool(False),
        FlagCHIPS   = cms.untracked.bool(False),
        FlagFTF     = cms.untracked.bool(False),
        FlagGlauber = cms.untracked.bool(False),
        FlagHP      = cms.untracked.bool(False),
        GFlash = cms.PSet(
        GflashHistogram = cms.bool(False),
        GflashEMShowerModel = cms.bool(False),
        GflashHadronPhysics = cms.string('QGSP_BERT_EMV'),
        GflashHadronShowerModel = cms.bool(False)
        )
        )
		process.g4SimHits.G4Commands = cms.vstring('/tracking/verbose 1')

	elif FLAVOR =="stau":
		process.g4SimHits.Physics = cms.PSet(
        process.customPhysicsSetup,
        DummyEMPhysics = cms.bool(True),
        G4BremsstrahlungThreshold = cms.double(0.5), ## cut in GeV    
        DefaultCutValue = cms.double(1.), ## cuts in cm,default 1cm    
        CutsPerRegion = cms.bool(True),
        Verbosity = cms.untracked.int32(0),
        type = cms.string('SimG4Core/Physics/CustomPhysics'),
        )
		process.g4SimHits.G4Commands = cms.vstring('/tracking/verbose 1')
	
	else:
		print "Wrong flavor %s. Only accepted are gluino, stau, stop." % FLAVOR

	return process
	
