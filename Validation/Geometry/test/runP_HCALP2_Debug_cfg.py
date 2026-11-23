###############################################################################
# Way to use this:
#   cmsRun runP_HCALP2_Debug_cfg.py geometry=D121
#
#   Options for geometry D104, D110, D111, D112, D113, D114, D115, D120, D121,
#                        D122, D123, D124, D125, D126, D127, D128, D129, D130
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

###############################################################################
#### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D128",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D104, D110, D111, D112, D113, D114, D115, D120, D121, D122, D123, D124, D125, D126, D127, D128, D129, D130")

### get and parse the command line arguments
options.parseArguments()

print(options)

###############################################################################
# Use the options

geomName = "Run4" + options.geometry
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
fileName = "matbdgHCAL" + geomName + "Debug.root"

print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)
print("o/p File Name:   ", fileName)

process = cms.Process('PROD',ERA)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load(geomFile)
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("GeneratorInterface.Core.generatorSmeared_cfi")
from Configuration.StandardSequences.VtxSmeared import VtxSmeared
process.load(VtxSmeared['NoSmear'])

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.VtxSmeared.engineName = cms.untracked.string('HepJamesRandom')
process.RandomNumberGeneratorService.generator.initialSeed = 456789
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876
process.RandomNumberGeneratorService.VtxSmeared.initialSeed = 123456789

process.load('FWCore.MessageService.MessageLogger_cfi')
if hasattr(process,'MessageLogger'):
    process.MessageLogger.MaterialBudget=dict()
#   process.MessageLogger.MaterialBudgetFull=dict()

process.source = cms.Source("EmptySource",
    firstRun        = cms.untracked.uint32(1),
    firstEvent      = cms.untracked.uint32(1)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        PartID = cms.vint32(14),
        MinEta = cms.double(2.8),
        MaxEta = cms.double(3.0),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(10.0),
        MaxE   = cms.double(10.0)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity       = cms.untracked.int32(0)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(fileName)
)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.LHCTransport = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Generator.ApplyEtaCuts = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetHcal = cms.PSet(
        FillHisto    = cms.untracked.bool(False),
        PrintSummary = cms.untracked.bool(True),
        DoHCAL       = cms.untracked.bool(True),
        TestVolumes  = cms.untracked.bool(True),
        NBinPhi      = cms.untracked.int32(180),
        NBinEta      = cms.untracked.int32(260),
        MaxEta       = cms.untracked.double(5.2),
        EtaLow       = cms.untracked.double(-3.0),
        EtaHigh      = cms.untracked.double(3.0),
        EtaMinP      = cms.untracked.double(-5.5),
        EtaMaxP      = cms.untracked.double(5.5),
        EtaLowMin    = cms.untracked.double(0.783),
        EtaLowMax    = cms.untracked.double(0.870),
        EtaMidMin    = cms.untracked.double(2.650),
        EtaMidMax    = cms.untracked.double(2.868),
        EtaHighMin   = cms.untracked.double(2.868),
        EtaHighMax   = cms.untracked.double(3.000),
        RMax         = cms.untracked.double(5.0),
        ZMax         = cms.untracked.double(14.0),
        Fromdd4hep   = cms.untracked.bool(False)
    ),
    type = cms.string('MaterialBudgetHcal')
))

# Schedule definition
process.p1 = cms.Path(process.generator*process.VtxSmeared*process.generatorSmeared*process.g4SimHits)
