import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

process = cms.Process("testHGCalRecoLocal",eras.Run2_25ns)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
#process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2023DevReco_cff')
process.load('Configuration.Geometry.GeometryExtended2023Dev_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedGauss_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

### setup HGCal local reco
# get uncalibrechits with weights method
process.load("RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi")
process.HGCalUncalibRecHit.HGCEEdigiCollection  = 'mix:HGCDigisEE'
process.HGCalUncalibRecHit.HGCHEFdigiCollection = 'mix:HGCDigisHEfront'
#process.HGCalUncalibRecHit.HGCHEBdigiCollection = 'mix:HGCDigisHEback'

# get rechits e.g. from the weights
process.load("RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi")
process.HGCalRecHit.HGCEEuncalibRecHitCollection  = 'HGCalUncalibRecHit:HGCEEUncalibRecHits'
process.HGCalRecHit.HGCHEFuncalibRecHitCollection = 'HGCalUncalibRecHit:HGCHEFUncalibRecHits'
#process.HGCalRecHit.HGCHEBuncalibRecHitCollection = 'HGCalUncalibRecHit:HGCHEBUncalibRecHits'


process.HGCalRecoLocal = cms.Sequence(process.HGCalUncalibRecHit +
                                      process.HGCalRecHit )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(500)
)

process.MessageLogger.cerr.FwkReport.reportEvery = 5
# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string(''),
    annotation = cms.untracked.string(''),
    name = cms.untracked.string('Applications')
)


# Output definition
process.output = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    outputCommands = cms.untracked.vstring(
        'drop *',
	#'keep *',
	#'keep *_genParticles_*_*',
	'keep *_g4SimHits_*_*',
        #'keep *_mix_*_*',
	'keep *_*HGC*_*_*',
        #'keep *_HGCalUncalibRecHit_*_*',
        #'keep *_HGCalRecHit_*_*'
        ),
    fileName = cms.untracked.string('file:testHGCalSimWatcher.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW-RECO')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    )
)

process.MessageLogger = cms.Service("MessageLogger",
    cout = cms.untracked.PSet(
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        ValidHGCal = cms.untracked.PSet(
            limit = cms.untracked.int32(-1)
        ),
    ),
    categories = cms.untracked.vstring(
        'ValidHGCal'),
    destinations = cms.untracked.vstring('cout','cerr')
)

# Additional output definition
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    SimG4HGCalValidation = cms.PSet(
        Names = cms.vstring(
		'HGCalEECell',  
		'HGCalHECell',
		'HEScintillator',
		),
	Types = cms.vint32(1,1,2),
	LabelLayerInfo = cms.string("HGCalInfoLayer"),
    ),
    type = cms.string('SimG4HGCalValidation')
))


#process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
        #Name           = cms.untracked.string('HCal*'),
#        Name           = cms.untracked.string('HGC*'),
#        type           = cms.string('PrintSensitive')))

#For Fast Timing
#"SensitiveDetector" value="FastTimerSensitiveDetector" eval="false"/>
#<Parameter name="ReadOutName" value="FastTimerHits"

# Other statements
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

process.generator = cms.EDProducer("FlatRandomPtGunProducer",
    PGunParameters = cms.PSet(
        MaxPt = cms.double(35.0),
        MinPt = cms.double(35.0),
        #PartID = cms.vint32(11), #--->electron
        PartID = cms.vint32(13), #--->muon
        #PartID = cms.vint32(211), #--->pion
        MaxEta = cms.double(2.9),
        MaxPhi = cms.double(3.14159265359),
        MinEta = cms.double(1.6),
        MinPhi = cms.double(-3.14159265359)
    ),
    Verbosity = cms.untracked.int32(0),
    psethack = cms.string('single muon pt 35'),
    AddAntiParticle = cms.bool(False),
    firstRun = cms.untracked.uint32(1)
)


#Modified to produce hgceedigis
#process.load('RecoLocalCalo.HGCalRecProducers.mixNoPU_cfi')
process.mix.digitizers = cms.PSet(process.theDigitizersValid)

#Following Removes Mag Field
process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.bField = cms.double(0.0)
# Define pileUp
#process.mix.input.nbPileupEvents.averageNumber = cms.double(100)


# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.recotest_step = cms.Path(process.HGCalRecoLocal)
process.out_step = cms.EndPath(process.output)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.digitisation_step,process.L1simulation_step,process.digi2raw_step,process.recotest_step,process.out_step)
# filter all path with the production filter sequence
for path in process.paths:
        getattr(process,path)._seq = process.generator * getattr(process,path)._seq

# customisation of the process.

# Automatic addition of the customisation function from SLHCUpgradeSimulations.Configuration.combinedCustoms
from SLHCUpgradeSimulations.Configuration.combinedCustoms import cust_2023HGCal

#call to customisation function cust_2023HGCalMuon imported from SLHCUpgradeSimulations.Configuration.combinedCustoms
process = cust_2023HGCal(process)

# End of customisation functions
