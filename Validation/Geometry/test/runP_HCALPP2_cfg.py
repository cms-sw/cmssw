###############################################################################
# Way to use this:
#   cmsRun runP_HCALPP2_cfg.py geometry=D121
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
fileName = "matbdgHCALP" + geomName + ".root"

print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)
print("o/p File Name:   ", fileName)

process = cms.Process('PROD',ERA)

process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load(geomFile)
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(10000)
if hasattr(process,'MessageLogger'):
    process.MessageLogger.MaterialBudget=dict()

process.source = cms.Source("PoolSource",
    noEventSort = cms.untracked.bool(True),
    duplicateCheckMode = cms.untracked.string("noDuplicateCheck"),
    fileNames = cms.untracked.vstring('file:single_neutrino_random.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string(fileName)
)

process.load("Validation.Geometry.materialBudgetHcalAnalysis_cfi")

process.simulation_step = cms.Path(process.g4SimHits)
process.analysis_step = cms.Path(process.materialBudgetHcalAnalysis)

process.g4SimHits.UseMagneticField = False
process.g4SimHits.LHCTransport = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetHcalProducer = cms.PSet(
        RMax         = cms.untracked.double(5.0),
        ZMax         = cms.untracked.double(14.0),
        Fromdd4hep   = cms.untracked.bool(False),
        EtaMinP      = cms.untracked.double(5.2),
        EtaMaxP      = cms.untracked.double(0.0),
        PrintSummary = cms.untracked.bool(True),
        Name         = cms.untracked.string('Hcal')
    ),
    type = cms.string('MaterialBudgetHcalProducer')
))

# Schedule definition
process.schedule = cms.Schedule(process.simulation_step,
                                process.analysis_step,
                                )
