import FWCore.ParameterSet.Config as cms

import random
import math

from Configuration.StandardSequences.Eras import eras
process = cms.Process('SIM',eras.Run2_2016)

# import of standard configurations
process.load("CondCore.CondDB.CondDB_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeV2016Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('Configuration.Geometry.GeometryExtended2016_CTPPS_cff')

process.RandomNumberGeneratorService.generator.initialSeed = cms.untracked.uint32(random.randint(0,900000000))

nEvent_ = 1000
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(nEvent_)
        )

process.source = cms.Source("EmptySource")
"""
process.source = cms.Source("EmptySource",
                #firstRun = cms.untracked.uint32(306572), # 2016H data
                #firstTime = cms.untracked.uint64(6487615523004612608)  # this is needed because it lacks the MC tag, run based
                #firstRun = cms.untracked.uint32(273730), # 2016H data
                #firstTime = cms.untracked.uint64(6286859745043152896)  # this is needed because it lacks the MC tag, run based
                firstRun = cms.untracked.uint32(282730), # 2016H data
                firstTime = cms.untracked.uint64(6339435345951588352)  # this is needed because it lacks the MC tag, run based
)
"""

process.options = cms.untracked.PSet()


from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
#process.GlobalTag = GlobalTag(process.GlobalTag, "106X_dataRun2_v26")

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('GeneratorInterface/ExhumeInterface/python/ExHuME_CEPDijetsGG_M100_7TeV_cff.py nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# generator

process.generator = cms.EDFilter("ExhumeGeneratorFilter",
    ExhumeParameters = cms.PSet(
        AlphaEw = cms.double(0.0072974),
        B = cms.double(4.0),
        BottomMass = cms.double(4.6),
        CharmMass = cms.double(1.42),
        HiggsMass = cms.double(120.0),
        HiggsVev = cms.double(246.0),
        LambdaQCD = cms.double(80.0),
        MinQt2 = cms.double(0.64),
        MuonMass = cms.double(0.1057),
        PDF = cms.double(11000),
        Rg = cms.double(1.2),
        StrangeMass = cms.double(0.19),
        Survive = cms.double(0.03),
        TauMass = cms.double(1.77),
        TopMass = cms.double(175.0),
        WMass = cms.double(80.33),
        ZMass = cms.double(91.187)
    ),
    ExhumeProcess = cms.PSet(
        MassRangeHigh = cms.double(2000.0),
        MassRangeLow = cms.double(300.0),
        ProcessType = cms.string('GG'),
        ThetaMin = cms.double(0.3)
    ),
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring()
    ),
    comEnergy = cms.double(13000.0),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1)
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

############
process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
        fileName = cms.untracked.string('gluglu_step1_SIM2016.root')
        )

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq

