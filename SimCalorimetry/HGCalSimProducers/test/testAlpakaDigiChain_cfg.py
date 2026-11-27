import FWCore.ParameterSet.Config as cms
process = cms.Process("TEST")

from FWCore.ParameterSet.VarParsing import VarParsing
options = VarParsing('python')
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

#source
process.maxEvents = cms.untracked.PSet( 
    input = cms.untracked.int32(options.maxEvents) 
)
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles)
)

# Geometry
process.load('Configuration.Geometry.GeometryExtendedRun4D122Reco_cff')

# electronics mapping
from Geometry.HGCalMapping.hgcalmapping_cff import customise_hgcalmapper
kwargs = { k: getattr(options,k) for k in ['modules','sicells','sipmcells','offsetfile'] if getattr(options,k)!='' }
process = customise_hgcalmapper(process, **kwargs)

# ageing conditions
from SimCalorimetry.HGCalSimAlgos.hgcal_ageing_byalgo_cfi import customise_hgcal_ageing_byalgo_startup
process = customise_hgcal_ageing_byalgo_startup(process)

# Geant4 accumulator: edmPluginHelp -m HGCalSimHitAccumulator@alpaka
from SimCalorimetry.HGCalSimProducers.hgCalSimHitAccumulator_cfi import hgCalSimHitAccumulator
process.hgCalSimHitAccumulator = cms.EDProducer(
    'alpaka_serial_sync::HGCalSimHitAccumulator',
    g4HitsDets = cms.vint32([8,9,10]),
    g4Hits = cms.VInputTag(
        cms.InputTag("g4SimHits", "HGCHitsEE"),
        cms.InputTag("g4SimHits", "HGCHitsHEfront"),
        cms.InputTag("g4SimHits", "HGCHitsHEback"),
    ),
    g4TOFDelays = cms.vdouble([-9,-11,-14]),
    indexingSource = cms.ESInputTag(),
    cellMappingSource = cms.ESInputTag()
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


process.RandomNumberGeneratorService = cms.Service(
    "RandomNumberGeneratorService",
    hgCalDigitizerProducer = cms.PSet(
        initialSeed = cms.untracked.uint32(1337),
        engineName = cms.untracked.string("HepJamesRandom")
    )
)

# test path
process.p = cms.Path(
    process.hgCalSimHitAccumulator
    *process.hgCalDigitizerProducer
)
