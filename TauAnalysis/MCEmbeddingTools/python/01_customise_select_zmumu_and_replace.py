import FWCore.ParameterSet.Config as cms

# cuts: ---------------------------------------------------------------------

runOnTheGrid = False

# Events to process
events_to_process=-1

TauolaNoPolar = cms.PSet(
	UseTauolaPolarization = cms.bool(False)
)
TauolaPolar = cms.PSet(
	UseTauolaPolarization = cms.bool(True)
)

from Configuration.Generator.PythiaUESettings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *
TauolaDefaultInputCards.InputCards.mdtau = cms.int32(102);

# Note: currently this is just a sketch and should not be used
from IOMC.RandomEngine.RandomServiceHelper import RandomNumberServiceHelper


# ------------------------------------------------------------------------------
			
def customise(process):
	if hasattr(process, "RandomNumberGeneratorService"):
		randSvc = RandomNumberServiceHelper(process.RandomNumberGeneratorService)
		randSvc.populate()
	
	process._Process__name="SELECTION"
	process.LoadAllDictionaries = cms.Service("LoadAllDictionaries")
	process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
	#process.Tracer = cms.Service("Tracer")
	
	process.TFileService = cms.Service("TFileService", 
		fileName = cms.string("histo.root"),
		closeFileFast = cms.untracked.bool(True)
	)
	
	process.load("ElectroWeakAnalysis.ZReco.dimuons_SkimPaths_cff")
	#process.load("ElectroWeakAnalysis.Skimming.dimuons_SkimPaths_cff")
	process.schedule.insert(len(process.schedule)-1,process.dimuonsPath)
	
	# Output module configuration
	#process.load("ElectroWeakAnalysis.Skimming.dimuonsOutputModule_cfi")
	
	process.load("TrackingTools.TrackAssociator.default_cfi")
	
	process.selectMuons = cms.EDProducer('SelectReplacementCandidates',
		process.TrackAssociatorParameterBlock,
		muonInputTag = cms.InputTag("muons")
	)

	process.prepareMuonsPath = cms.Path(process.selectMuons)
	process.schedule.insert(len(process.schedule)-1,process.prepareMuonsPath)
	process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
	process.RandomNumberGeneratorService.newSource = cms.PSet(
		initialSeed = cms.untracked.uint32(12345),
		engineName = cms.untracked.string('HepJamesRandom')
	)                                                                                               
	process.load("IOMC/RandomEngine/IOMC_cff")
	
	process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")
	process.RandomNumberGeneratorService.newSource = cms.PSet(
		      initialSeed = cms.untracked.uint32(12345),
		      engineName = cms.untracked.string('HepJamesRandom')
	) 
	process.RandomNumberGeneratorService.theSource = cms.PSet(
		      initialSeed = cms.untracked.uint32(12345),
		      engineName = cms.untracked.string('HepJamesRandom')
	)
			    
	process.newSource = cms.EDProducer("MCParticleReplacer",
    src                = cms.InputTag("muons"),
    beamSpotSrc        = cms.InputTag("dummy"),
    primaryVertexLabel = cms.InputTag("dummy"),
    hepMcSrc           = cms.InputTag("generator"),

    algorithm = cms.string("ZTauTau"), # "ParticleGun", "ZTauTau", "CommissioningGun"
    hepMcMode = cms.string("new"),         # "new" for new HepMCProduct with taus and decay products,
                                           # "replace" for replacing muons in the existing HepMCProcuct
                                           # commissioning
    verbose = cms.bool(False),

                CommissioningGun = cms.PSet(
                        maxMuonEta = cms.double(2.1),
                        minMuonPt = cms.double(5.)
                ),


    ZTauTau = cms.PSet(
		    TauolaOptions = cms.PSet(
		  		TauolaPolar,
		  		InputCards = cms.PSet
					(
					    pjak1 = cms.int32(0),
		    			pjak2 = cms.int32(0),
					    mdtau = cms.int32(102)
		 			)
		 		),
        filterEfficiency = cms.untracked.double(1.0),
        pythiaHepMCVerbosity = cms.untracked.bool(False),
        generatorMode = cms.string("Tauola"),  # "Tauola", "Pythia" (not implemented yet)
        
    )
	)
	
	
	process.insertNewSourcePath = cms.Path(process.newSource)
	process.schedule.insert(len(process.schedule)-1,process.insertNewSourcePath)
                                              
	process.options = cms.untracked.PSet( SkipEvent = cms.untracked.vstring('ProductNotFound') )

	if runOnTheGrid:
		process.source.fileNames=cms.untracked.vstring(__FILE_NAMES__)
		process.source.skipEvents=cms.untracked.uint32(__SKIP_EVENTS__)
		process.maxEvents.input = cms.untracked.int32(__MAX_EVENTS__)
		process.output.fileName=cms.untracked.string("output.root")

	process.filterNumHepMCEvents = cms.EDFilter('EmptyEventsFilter',
		minEvents=cms.untracked.int32(2),
		target=cms.untracked.int32(1)
	)		
	process.filterNumHepMCEventsPath = cms.Path(process.filterNumHepMCEvents)
	process.schedule.insert(len(process.schedule)-1,process.filterNumHepMCEventsPath)
		
	process.output.SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('filterNumHepMCEventsPath'))	
	process.output.outputCommands = cms.untracked.vstring(
                "drop *_*_*_*",
								"keep edmHepMCProduct_*_*_*",
                "keep CaloTowersSorted_*_*_*",
                "keep recoMuons_*_*_*",
                "keep recoCaloMETs_met_*_*",
                "keep *_overlay_*_*",
                "keep *_selectMuons_*_*",
                "keep *_selectMuonsForMuonMuonReplacement_*_*",                
                "keep EBDigiCollection_*_*_*",
                "keep EEDigiCollection_*_*_*",
                "keep ESDataFramesSorted_*_*_*",
                "keep DTLayerIdDTDigiMuonDigiCollection_*_*_*",
                "keep CSCDetIdCSCStripDigiMuonDigiCollection_*_*_*",
                "keep CSCDetIdCSCWireDigiMuonDigiCollection_*_*_*",
                "keep CSCDetIdCSCComparatorDigiMuonDigiCollection_*_*_*",
                "keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*",
                "keep HBHEDataFramesSorted_*_*_*",
                "keep HFDataFramesSorted_*_*_*",
                "keep HODataFramesSorted_*_*_*",
                "keep *_hcalDigis_*_*",
                "keep SiStripDigiedmDetSetVector_*_*_*",
                "keep PixelDigiedmDetSetVector_*_*_*"
        )
	#print process.schedule
	print process.dumpPython()
	return(process)
