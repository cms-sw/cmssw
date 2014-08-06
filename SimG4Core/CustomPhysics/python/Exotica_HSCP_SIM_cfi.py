import FWCore.ParameterSet.Config as cms

def customise(process):
    
	FLAVOR = process.generator.hscpFlavor.value()
	MASS_POINT = process.generator.massPoint.value()
	SLHA_FILE = process.generator.slhaFile.value()
	PROCESS_FILE = process.generator.processFile.value()
	PARTICLE_FILE = process.generator.particleFile.value()
	PDT_FILE = process.generator.pdtFile.value()
	USE_REGGE = process.generator.useregge.value()

	process.load("SimG4Core.CustomPhysics.CustomPhysics_cfi")
	process.customPhysicsSetup.particlesDef = PARTICLE_FILE
	process.customPhysicsSetup.reggeModel = USE_REGGE
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
	process.HepPDTESSource.pdtFileName= PDT_FILE
	
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

	GflashEcal = cms.bool(False),
	bField = cms.double(3.8),
	energyScaleEB = cms.double(1.032),
	energyScaleEE = cms.double(1.024),
	GflashHcal = cms.bool(False),
	RusRoGammaEnergyLimit = cms.double(0.0),
	RusRoEcalGamma = cms.double(1.0),
	RusRoHcalGamma = cms.double(1.0),
	RusRoQuadGamma = cms.double(1.0),
	RusRoMuonIronGamma = cms.double(1.0),
	RusRoPreShowerGamma = cms.double(1.0),
	RusRoCastorGamma = cms.double(1.0),
	RusRoBeamPipeOutGamma = cms.double(1.0),
	RusRoWorldGamma = cms.double(1.0),
	RusRoElectronEnergyLimit = cms.double(0.0),
	RusRoEcalElectron = cms.double(1.0),
	RusRoHcalElectron = cms.double(1.0),
	RusRoQuadElectron = cms.double(1.0),
	RusRoMuonIronElectron = cms.double(1.0),
	RusRoPreShowerElectron = cms.double(1.0),
	RusRoCastorElectron = cms.double(1.0),
	RusRoBeamPipeOutElectron = cms.double(1.0),
	RusRoWorldElectron = cms.double(1.0),

	GFlash = cms.PSet(
	GflashHistogram = cms.bool(False),
        GflashEMShowerModel = cms.bool(False),
        GflashHadronPhysics = cms.string('QGSP_BERT_EMV'),
        GflashHadronShowerModel = cms.bool(False)
        )
        )
#		process.g4SimHits.G4Commands = cms.vstring('/tracking/verbose 1')

	elif FLAVOR =="stau":
		process.g4SimHits.Physics = cms.PSet(
        process.customPhysicsSetup,
        DummyEMPhysics = cms.bool(True),
        G4BremsstrahlungThreshold = cms.double(0.5), ## cut in GeV    
        DefaultCutValue = cms.double(1.), ## cuts in cm,default 1cm    
        CutsPerRegion = cms.bool(True),
        Verbosity = cms.untracked.int32(0),

	GflashEcal = cms.bool(False),
	bField = cms.double(3.8),
	energyScaleEB = cms.double(1.032),
	energyScaleEE = cms.double(1.024),
	GflashHcal = cms.bool(False),
	RusRoGammaEnergyLimit = cms.double(0.0),
	RusRoEcalGamma = cms.double(1.0),
	RusRoHcalGamma = cms.double(1.0),
	RusRoQuadGamma = cms.double(1.0),
	RusRoMuonIronGamma = cms.double(1.0),
	RusRoPreShowerGamma = cms.double(1.0),
	RusRoCastorGamma = cms.double(1.0),
	RusRoBeamPipeOutGamma = cms.double(1.0),
	RusRoWorldGamma = cms.double(1.0),
	RusRoElectronEnergyLimit = cms.double(0.0),
	RusRoEcalElectron = cms.double(1.0),
	RusRoHcalElectron = cms.double(1.0),
	RusRoQuadElectron = cms.double(1.0),
	RusRoMuonIronElectron = cms.double(1.0),
	RusRoPreShowerElectron = cms.double(1.0),
	RusRoCastorElectron = cms.double(1.0),
	RusRoBeamPipeOutElectron = cms.double(1.0),
	RusRoWorldElectron = cms.double(1.0),

	type = cms.string('SimG4Core/Physics/CustomPhysics'),
        )
	#	process.g4SimHits.G4Commands = cms.vstring('/tracking/verbose 1')
	
	else:
		print "Wrong flavor %s. Only accepted are gluino, stau, stop." % FLAVOR

	return process
	
