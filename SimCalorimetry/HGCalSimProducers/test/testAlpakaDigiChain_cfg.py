import FWCore.ParameterSet.Config as cms

#argument parsing
from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('standard')
try:
    from HGCalCommissioning.NanoTools.configure_sysval_nano_cff import setArgParserForNANO
    setArgParserForNANO(options)
except Exception as e:
    print(e)
    print('Was not able to configure NANO')
options.register('verbosity',0,mytype=VarParsing.varType.int,
                 info='Tester verbosity: 0 = Base+FED prints / 1 = +Module prints / 2 = +Cell prints')
options.register('modules','Geometry/HGCalMapping/data/ModuleMaps/modulelocator_test.txt',mytype=VarParsing.varType.string,
                 info="Path to module mapper. Absolute, or relative to CMSSW src directory")
options.register('sicells','Geometry/HGCalMapping/data/CellMaps/WaferCellMapTraces.txt',mytype=VarParsing.varType.string,
                 info="Path to Si cell mapper. Absolute, or relative to CMSSW src directory")
options.register('sipmcells','Geometry/HGCalMapping/data/CellMaps/channels_sipmontile.hgcal.txt',mytype=VarParsing.varType.string,
                 info="Path to SiPM-on-tile cell mapper. Absolute, or relative to CMSSW src directory")
options.register('offsetfile','Geometry/HGCalMapping/data/CellMaps/calibration_to_surrounding_offsetMap.txt',mytype=VarParsing.varType.string,
                 info="Path to calibration-to-surrounding cell offset file. Absolute, or relative to CMSSW src directory")
options.parseArguments()

#era
from Configuration.Eras.Era_Phase2C26I13M9_cff import Phase2C26I13M9
process = cms.Process("TEST")

# geometry
process.load('Configuration.Geometry.GeometryExtendedRun4D128Reco_cff')
process.load('Configuration.Geometry.GeometryExtendedRun4D128_cff')

#event content
process.load("Configuration.EventContent.EventContent_cff")

#source
process.maxEvents = cms.untracked.PSet( 
    input = cms.untracked.int32(options.maxEvents) 
)
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(options.files)
)

# electronics configuration
process.hgcalConfigESProducer = cms.ESSource( # ESProducer to load configurations for unpacker
    'HGCalConfigurationESProducer',
    fedjson=cms.FileInPath("HGCalCommissioning/Calibrations/P5/config/config_feds_v1.json"), # JSON with FED configuration parameters
    modjson=cms.FileInPath("HGCalCommissioning/Calibrations/P5/config/config_econds_v1.json"), # JSON with ECON-D configuration parameters
    bePassthroughMode=cms.int32(-1),
    cbHeaderMarker=cms.int32(-1),
    slinkHeaderMarker=cms.int32(-1),
    econdHeaderMarker=cms.int32(-1),
    charMode=cms.int32(-1),
    indexSource=cms.ESInputTag('hgCalMappingESProducer','')
)


# electronics mapping
from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
kwargs = { k: getattr(options,k) for k in ['modules','sicells','sipmcells','offsetfile'] if getattr(options,k)!='' }
process = customise_hgcalmapper(process, **kwargs)

# ageing conditions
from SimCalorimetry.HGCalSimAlgos.hgcal_ageing_byalgo_cfi import customise_hgcal_ageing_byalgo_startup
process = customise_hgcal_ageing_byalgo_startup(process)

# random number generator service
process.RandomNumberGeneratorService = cms.Service(
    "RandomNumberGeneratorService",
    hgCalDigitizerProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(1337),
        engineName = cms.untracked.string("HepJamesRandom")
    )
)

# NANO output (adds hgCalSimHitAccumTable + NanoAODOutputModule, preserves the paths above)
try:
    from HGCalCommissioning.NanoTools.configure_sysval_nano_cff import configureNANOStep
    options.skipMeta = True
    options.isSimulation = True
    process = configureNANOStep(process, options)
except Exception as e:
    print(e)
    print('NANO will not be produced')


# DIGI STEP

# Geant4 accumulator: edmPluginHelp -m HGCalSimHitAccumulator@alpaka
from SimCalorimetry.HGCalSimProducers.hgCalSimHitAccumulator_cfi import hgCalSimHitAccumulator
process.hgCalSimHitAccumulator = cms.EDProducer(
    'alpaka_serial_sync::HGCalSimHitAccumulator',
    g4HitsConfig = cms.VPSet([
        cms.PSet(
            det = cms.int32(8),
            iscalib = cms.bool(False),
            hits = cms.InputTag("g4SimHits", "HGCHitsEE"),
            tof_delay = cms.double(-9)
        ),
        cms.PSet(
            det = cms.int32(8),
            iscalib = cms.bool(True),
            hits = cms.InputTag("g4SimHits", "CalibrationHGCHitsEE", "SIM"),
            tof_delay = cms.double(-9)
        ),        
        cms.PSet(
            det = cms.int32(9),
            iscalib = cms.bool(False),
            hits = cms.InputTag("g4SimHits", "HGCHitsHEfront"),
            tof_delay = cms.double(-11)
        ),
        cms.PSet(
            det = cms.int32(9),
            iscalib = cms.bool(True),
            hits = cms.InputTag("g4SimHits", "CalibrationHGCHitsHEfront"),
            tof_delay = cms.double(-11)
        ),
        cms.PSet(
            det = cms.int32(10),
            iscalib = cms.bool(False),
            hits = cms.InputTag("g4SimHits", "HGCHitsHEback"),
            tof_delay = cms.double(-14)
        )
    ]),
    indexingSource = cms.ESInputTag(),
    cellMappingSource = cms.ESInputTag(),
    produceExtra = cms.bool(True)
)

# HGCal Alpaka digitizer (notice it's declaring the CPU version of it!)
from SimCalorimetry.HGCalSimProducers.hgCalDigitizerProducer_cfi import hgCalDigitizerProducer
process.hgCalDigitizerProducer = cms.EDProducer(
    'alpaka_serial_sync::HGCalDigitizerProducer',
    g4HitsAcc = cms.InputTag("hgCalSimHitAccumulator"),
    ageingConds = cms.ESInputTag(),
    n_blocks = cms.int32(4096),
    n_threads = cms.int32(1024)
)

process.digi_step = cms.Path(
    process.hgCalSimHitAccumulator
    *process.hgCalDigitizerProducer
)
try:
    process.schedule.insert(0, process.digi_step)
except:
    #NANO is not being used, start again
    process.schedule = cms.Schedule(process.digi_step)

# output file (standard EDM)
process.edmoutput = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_hgCalSimHitAccumulator_*_*',
        'keep *_hgCalDigitizerProducer_*_*'
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('digi_step')
    ),
    fileName = cms.untracked.string('DIGIs.root')
)
process.edmoutpath = cms.EndPath(process.edmoutput)
process.schedule.append( process.edmoutpath )


#DQM
if True:
    process.DQMStore = cms.Service("DQMStore")
    process.load("DQM.HGCAL.hgcalDQM_cff")
    process.hgcaldigidqm.Digis = cms.InputTag('hgCalDigitizerProducer','HGCROCDigis')
    process.load("DQMServices.FileIO.DQMFileSaverOnline_cfi")
    process.dqmSaver.tag = 'HGCAL'
    process.dqmSaver.runNumber = 1
    
    process.dqmPath = cms.Path(
        process.hgcalDQMSources  
        #+process.hgCalDQMHarvester 
        +process.dqmSaver
    )
    process.schedule.append( process.dqmPath )
    process.options.numberOfConcurrentLuminosityBlocks = 1
