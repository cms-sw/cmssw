import FWCore.ParameterSet.Config as cms
import math

process = cms.Process("RPDigiProducerTest")

# Specify the maximum events to simulate
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Configure the output module (save the result in a file)
# Configure the output module
process.o1 = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('file:RPDigiProducerTest_output.root')
)


process.load("SimGeneral.HepPDTESSource.pdt_cfi")


################## STEP 1
process.source = cms.Source("EmptySource")

################## STEP 2 - process.generator
process.load('Configuration.StandardSequences.Generator_cff')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789
process.RandomNumberGeneratorService.LHCTransport.engineName   = cms.untracked.string('TRandom3')
phi_min = -math.pi
phi_max = math.pi
t_min = 0.
t_max = 2.0
xi_min = 0.01
xi_max = 0.2
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

################## STEP 3 process.SmearingGenerator

# declare optics parameters

# Smearing
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic25ns13TeVEarly2017Collision_cfi')

################## STEP 4 process.OptInfo


################## STEP 5 process.*process.g4SimHits

# Magnetic Field, by default we have 3.8T
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')
process.load('PhysicsTools.HepMCCandAlgos.genParticles_cfi')

# G4 simulation & proton transport

from Geometry.VeryForwardGeometry.geometryPPS_CMSxz_fromDD_2017_cfi import XMLIdealGeometryESSource_CTPPS
process.XMLIdealGeometryESSource = XMLIdealGeometryESSource_CTPPS.clone()

process.load("SimG4Core.Application.g4SimHits_cfi")

process.g4SimHits.Generator.HepMCProductLabel = 'LHCTransport'    # The input source for G4 module is connected to "process.source".

################## STEP 6 process.mix*process.RPSiDetDigitizer 

# No pile up for the mixing module
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

########################### DIGI+RECO RP ##########################################

# process.load("SimPPS.RPDigiProducer.RPSiDetConf_cfi")
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.g4Simhits_step = cms.Path(process.g4SimHits)


process.simulation_step = cms.Path(process.psim)
process.g4Simhits_step = cms.Path(process.g4SimHits)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)

process.outpath = cms.EndPath(process.o1)

process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.g4Simhits_step,process.outpath)

# filter all path with the production filter sequence
for path in process.paths:
    getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq


# print process.dumpConfig()
from SimPPS.PPSSimTrackProducer.SimTrackProducerForFullSim_cff import customise
process = customise(process)

