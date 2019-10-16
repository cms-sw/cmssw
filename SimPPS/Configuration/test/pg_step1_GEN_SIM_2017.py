import FWCore.ParameterSet.Config as cms

import math

from Configuration.StandardSequences.Eras import eras
process = cms.Process('SIM',eras.Run2_2017,eras.ctpps_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeVEarly2017Collision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
############### using only CTPPS geometry 
from Geometry.VeryForwardGeometry.geometryPPS_CMSxz_fromDD_2017_cfi import XMLIdealGeometryESSource_CTPPS
process.XMLIdealGeometryESSource = XMLIdealGeometryESSource_CTPPS.clone()

process.load("SimG4Core.Application.g4SimHits_cfi")
process.g4SimHits.Generator.ApplyPCuts          = False
process.g4SimHits.Generator.ApplyPhiCuts        = False
process.g4SimHits.Generator.ApplyEtaCuts        = False
process.g4SimHits.Generator.HepMCProductLabel   = 'LHCTransport'
process.g4SimHits.Generator.MinEtaCut        = -13.0
process.g4SimHits.Generator.MaxEtaCut        = 13.0
process.g4SimHits.Generator.Verbosity        = 0

process.g4SimHits.G4TrackingManagerVerbosity = cms.untracked.int32(3)
process.g4SimHits.SteppingAction.MaxTrackTime = cms.double(2000.0)
process.g4SimHits.StackingAction.MaxTrackTime = cms.double(2000.0)

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.RandomNumberGeneratorService.LHCTransport.engineName   = cms.untracked.string('TRandom3')

nEvent_ = 1000
process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(nEvent_)
        )

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')
process.load('PhysicsTools.HepMCCandAlgos.genParticles_cfi')

phi_min = -math.pi
phi_max = math.pi
t_min   = 0.
t_max   = 2.
xi_min  = 0.01
xi_max  = 0.3
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

process.source = cms.Source("EmptySource",
)

process.ProductionFilterSequence = cms.Sequence(process.generator)

############
process.o1 = cms.OutputModule("PoolOutputModule",
        outputCommands = cms.untracked.vstring('keep *'),
        fileName = cms.untracked.string('step1_SIM2017.root')
        )

process.common_maximum_timex = cms.PSet( # need to be localy redefined
        MaxTrackTime  = cms.double(2000.0),  # need to be localy redefined
        MaxTimeNames  = cms.vstring('ZDCRegion'), # need to be localy redefined
        MaxTrackTimes = cms.vdouble(10000.0),  # need to be localy redefined
        DeadRegions = cms.vstring()
        )
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.g4Simhits_step = cms.Path(process.g4SimHits)


process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)

process.outpath = cms.EndPath(process.o1)

process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.g4Simhits_step,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq

from SimPPS.PPSSimTrackProducer.SimTrackProducerForFullSim_cff import customise
process = customise(process)

