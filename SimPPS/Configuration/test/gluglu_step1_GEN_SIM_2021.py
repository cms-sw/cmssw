import FWCore.ParameterSet.Config as cms

import random
import math

from Configuration.StandardSequences.Eras import eras
process = cms.Process('SIM',eras.Run3)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')

process.RandomNumberGeneratorService.generator.initialSeed = cms.untracked.uint32(random.randint(0,900000000))

process.load('SimG4Core.Application.g4SimHits_cfi')
process.g4SimHits.LHCTransport = cms.bool(True)

nEvent_ = 100
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(nEvent_)
        )

process.source = cms.Source("EmptySource")

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2022_realistic', '')

# generator
process.generator = cms.EDFilter("ExhumeGeneratorFilter",
    ExhumeParameters = cms.PSet(
        AlphaEw = cms.double(0.0072974),
        B = cms.double(4.0),
        MuonMass = cms.double(0.1057),
        BottomMass = cms.double(4.65),
        CharmMass = cms.double(1.28),
        StrangeMass = cms.double(0.095),
        TauMass = cms.double(1.78),
        TopMass = cms.double(172.8),
        WMass = cms.double(80.38),
        ZMass = cms.double(91.187),
        HiggsMass = cms.double(125.1),
        HiggsVev = cms.double(246.0),
        LambdaQCD = cms.double(80.0),
        MinQt2 = cms.double(0.64),
        PDF = cms.double(11000),
        Rg = cms.double(1.2),
        Survive = cms.double(0.03)
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
    comEnergy = cms.double(14000.0),
    maxEventsToPrint = cms.untracked.int32(2),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(1)
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('GluGluTo2Jets_M_100_7TeV_exhume_cff.py nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

############
process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
        fileName = cms.untracked.string('GluGlu_step1_GEN_SIM_2021.root')
        )

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)

process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq

