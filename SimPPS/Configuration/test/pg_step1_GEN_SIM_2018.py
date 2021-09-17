import FWCore.ParameterSet.Config as cms

import random
import math

from Configuration.StandardSequences.Eras import eras
process = cms.Process('SIM',eras.Run2_2018)

# import of standard configurations
process.load("CondCore.CondDB.CondDB_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('Configuration.Geometry.GeometryExtended2018_CTPPS_cff')

process.RandomNumberGeneratorService.generator.initialSeed = cms.untracked.uint32(random.randint(0,900000000))

nEvent_ = 1000
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(nEvent_)
        )

process.source = cms.Source("EmptySource")
"""
process.source = cms.Source("EmptySource",
                firstRun = cms.untracked.uint32(315252),                # 2018A
                firstTime = cms.untracked.uint64(6548822380385402880)
                #firstRun = cms.untracked.uint32(319337),               # 2018C
                #firstTime = cms.untracked.uint64(6575656846424539136)
                #firstRun = cms.untracked.uint32(323363),               # 2018D
                #firstTime = cms.untracked.uint64(6604361306864615424)  
                #firstRun = cms.untracked.uint32(324612),               #2018D
                #firstTime = cms.untracked.uint64(6612348794983940096)  
)
"""


process.options = cms.untracked.PSet()

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2018_realistic', '')


# generator

phi_min = -math.pi
phi_max = math.pi
t_min   = 0.
t_max   = 2.
xi_min  = 0.02
xi_max  = 0.2
ecms = 13000.

process.generator = cms.EDProducer("RandomtXiGunProducer",
        PGunParameters = cms.PSet(
            PartID = cms.vint32(2212),
            MinPhi = cms.double(phi_min),
            MaxPhi = cms.double(phi_max),
            ECMS   = cms.double(ecms),
            Mint   = cms.double(t_min),
            Maxt   = cms.double(t_max),
            MinXi  = cms.double(xi_min),
            MaxXi  = cms.double(xi_max)
            ),
        Verbosity = cms.untracked.int32(0),
        psethack = cms.string('single protons'),
        FireBackward = cms.bool(True),
        FireForward  = cms.bool(True),
        firstRun = cms.untracked.uint32(1),
        )


process.ProductionFilterSequence = cms.Sequence(process.generator)

############
process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
        fileName = cms.untracked.string('step1_SIM2018.root')
        )

process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)


process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.outpath = cms.EndPath(process.o1)
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq

