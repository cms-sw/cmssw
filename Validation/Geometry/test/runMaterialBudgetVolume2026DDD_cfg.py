###############################################################################
# Way to use this:
#   cmsRun runMaterialBudgetVolume2026DDD_cfg.py geometry=D92
#
#   Options for geometry D86, D88, D91, D92, D93, D94, D95, D96, D97, D98,
#                        D99, D100, D101, D102, D103, D104, D105, D106, D107,
#                        D108, D109, D110
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D86, D88, D91, D92, D93, D94, D95, D96, D97, D98, D99, D100, D101, D102, D103, D104, D105, D106, D107, D108, D109, D110")
### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D94"):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    process = cms.Process('MaterialBudgetVolume',Phase2C20I13M9)
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('MaterialBudgetVolume',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
fileName = "matbdg" + options.geometry + "DDD" + ".root"

print("Geometry file Name: ", geomFile)
print("Root file Name:     ", fileName)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")
process.load("Validation.Geometry.materialBudgetVolume_cfi")

process.load("IOMC.RandomEngine.IOMC_cff")
process.RandomNumberGeneratorService.g4SimHits.initialSeed = 9876

process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)
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

process.g4SimHits.UseMagneticField = False
process.g4SimHits.Physics.type = 'SimG4Core/Physics/DummyPhysics'
process.g4SimHits.StackingAction.TrackNeutrino = True
process.g4SimHits.Physics.DummyEMPhysics = True
process.g4SimHits.Physics.CutsPerRegion = False

process.load("Validation.Geometry.materialBudgetVolumeAnalysis_cfi")
process.p1 = cms.Path(process.g4SimHits+process.materialBudgetVolumeAnalysis)
