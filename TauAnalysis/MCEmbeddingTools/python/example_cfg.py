import FWCore.ParameterSet.Config as cms
process = cms.Process("EXAMPLE")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.load("Configuration.Generator.PythiaUESettings_cfi")

# the following lines are required by hit tracking
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Geometry.CaloEventSetup.CaloTopology_cfi")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.GlobalTag.globaltag = 'IDEAL_V9::All'


TauolaDefaultInputCards = cms.PSet(
    InputCards = cms.vstring('TAUOLA = 0 0 102 ! TAUOLA ')      # 114=l+jet, 102=only muons
)
TauolaNoPolar = cms.PSet(
    UseTauolaPolarization = cms.bool(False)
)
TauolaPolar = cms.PSet(
    UseTauolaPolarization = cms.bool(True)
)

process.newSource = cms.EDProducer('MCParticleReplacer',
	desiredReplacerClass=cms.untracked.int32(1),
# replacementMode =
#    0 - remove Myons from existing HepMCProduct and implant taus (+decay products)
#    1 - build new HepMCProduct only with taus (+decay products)
	replacementMode  = cms.untracked.int32(1),
#
# generatorMode =
#    0 - use Pythia
#    1 - use Tauola  
	generatorMode  = cms.untracked.int32(1),

	printEvent = cms.untracked.bool(True),

    #maxEventsToPrint = cms.untracked.int32(5),
    #pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    ExternalGenerators = cms.PSet(
        Tauola = cms.untracked.PSet(
            TauolaPolar,
            TauolaDefaultInputCards
        ),
        parameterSets = cms.vstring('Tauola')
    ),
    UseExternalGenerators = cms.untracked.bool(True),
    PythiaParameters = cms.PSet(
        process.pythiaUESettingsBlock,
        ZtautauParameters = cms.vstring('MSEL         = 11 ', 
            'MDME( 174,1) = 0    !Z decay into d dbar', 
            'MDME( 175,1) = 0    !Z decay into u ubar', 
            'MDME( 176,1) = 0    !Z decay into s sbar', 
            'MDME( 177,1) = 0    !Z decay into c cbar', 
            'MDME( 178,1) = 0    !Z decay into b bbar', 
            'MDME( 179,1) = 0    !Z decay into t tbar', 
            'MDME( 182,1) = 0    !Z decay into e- e+', 
            'MDME( 183,1) = 0    !Z decay into nu_e nu_ebar', 
            'MDME( 184,1) = 0    !Z decay into mu- mu+', 
            'MDME( 185,1) = 0    !Z decay into nu_mu nu_mubar', 
            'MDME( 186,1) = 1    !Z decay into tau- tau+', 
            'MDME( 187,1) = 0    !Z decay into nu_tau nu_taubar', 
            'CKIN( 1)     = 40.  !(D=2. GeV)', 
            'CKIN( 2)     = -1.  !(D=-1. GeV)',
                'MSTJ(28) = 1          ! no tau decays in pythia, use tauola instead'),
        parameterSets = cms.vstring('pythiaUESettings', 
            'ZtautauParameters')
    )
)

process.selectMuons = cms.EDProducer('SelectParticles',
	TrackAssociatorParameters = cms.PSet(
			muonMaxDistanceSigmaX = cms.double(0.0),
			muonMaxDistanceSigmaY = cms.double(0.0),
			CSCSegmentCollectionLabel = cms.InputTag("cscSegments"),
			dRHcal = cms.double(9999.0),
			dREcal = cms.double(9999.0),
			CaloTowerCollectionLabel = cms.InputTag("towerMaker"),
			useEcal = cms.bool(True),
			dREcalPreselection = cms.double(0.05),
			HORecHitCollectionLabel = cms.InputTag("horeco"),
			dRMuon = cms.double(9999.0),
			crossedEnergyType = cms.string('SinglePointAlongTrajectory'),
			propagateAllDirections = cms.bool(True),
			muonMaxDistanceX = cms.double(5.0),
			muonMaxDistanceY = cms.double(5.0),
			useHO = cms.bool(True),
			accountForTrajectoryChangeCalo = cms.bool(False),
			DTRecSegment4DCollectionLabel = cms.InputTag("dt4DSegments"),
			EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEE"),
			dRHcalPreselection = cms.double(0.2),
			useMuon = cms.bool(True),
			useCalo = cms.bool(False),
			EBRecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEB"),
			dRMuonPreselection = cms.double(0.2),
			truthMatch = cms.bool(False),
			HBHERecHitCollectionLabel = cms.InputTag("hbhereco"),
			useHcal = cms.bool(True)
	),
	muonInputTag = cms.InputTag("muons")
)

process.source = cms.Source("PoolSource",
        skipBadFiles = cms.untracked.bool(True),
        skipEvents = cms.untracked.uint32(0),
        fileNames = cms.untracked.vstring('file:/home/elpis/ekp/events/copy_Zmumu_Summer08_IDEAL_V9_v1_GEN-SIM-RECO_2008-12-29_0.root')
)

process.load("Configuration.EventContent.EventContent_cff")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'DEBUG'

process.OUTPUT = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring("keep *_*_*_*"),
        SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('p1')),
        fileName = cms.untracked.string('grid_job_output.root')
)


process.dump = cms.EDAnalyzer("EventContentAnalyzer")

#process.raw2digi_step = cms.Path(process.RawToDigi)
#process.reconstruction_step = cms.Path(process.reconstruction)

process.p1 = cms.Path(process.selectMuons*process.newSource)
#process.p1 = cms.Path(process.RawToDigi*process.reconstruction*process.selectMuonHits)

#process.outpath = cms.EndPath(process.OUTPUT)
#



