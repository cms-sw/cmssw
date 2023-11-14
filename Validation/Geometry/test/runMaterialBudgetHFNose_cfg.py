###############################################################################
# Way to use this:
#   cmsRun runMaterialBudgetHFNose_cfg.py geometry=D94 type=DD4hep pos=Start
#
#   Options for geometry D92, D94
#   Options for type DD4hep, DDD
#   Options for pos Start, End
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D94",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D92, D94")
options.register('type',
                 "DDD",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: DDD, DD4hep")
options.register('pos',
                 "Start",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: Start, End")
### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D94"):
    nose = 1
else:
    nose = 0

if (options.type == "DD4hep"):
    flag = "DD4hep"
    ddFlag = True
else:
    flag = ""
    ddFlag = False

if (options.pos == "End"):
    zMax = 10.560
    tag = "End"
else:
    zMax = 9.6072
    tag = "Front"

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
if (nose == 1):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    if (ddFlag == True):
        process = cms.Process('MaterialBudgetVolume',Phase2C20I13M9,dd4hep)
    else:
        process = cms.Process('MaterialBudgetVolume',Phase2C20I13M9)
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    if (ddFlag == True):
        process = cms.Process('MaterialBudgetVolume',Phase2C17I13M9,dd4hep)
    else:
        process = cms.Process('MaterialBudgetVolume',Phase2C17I13M9)

geomFile = "Configuration.Geometry.Geometry" + flag + "Extended2026" + options.geometry + "Reco_cff"
fileName = "matbdgHFNose" + flag + options.geometry + tag + ".root"

print("Geometry file Name: ", geomFile)
print("Root file Name:     ", fileName)
print("nose Flag:          ", nose)
print("ddFlag Flag:        ", ddFlag)
print("zMax (m):           ", zMax)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("SimG4Core.Application.g4SimHits_cfi")

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
process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    MaterialBudgetVolume = cms.PSet(
        lvNames = cms.vstring('BEAM', 'BEAM1', 'BEAM2', 'BEAM3', 'BEAM4', 'Tracker', 'ECAL', 'HCal', 'VCAL', 'MGNT', 'MUON', 'OQUA', 'CALOEC', 'HFNoseVol'),
        lvLevels = cms.vint32(3, 3, 3, 3, 3, 3, 4, 4, 3, 4, 3, 3, 4, 3),
        useDD4hep = cms.bool(ddFlag),
        rMax = cms.double(1.8),
        zMax = cms.double(zMax),
    ),
    type = cms.string('MaterialBudgetVolume'),
))

process.load("Validation.Geometry.materialBudgetVolumeAnalysis_cfi")
process.p1 = cms.Path(process.g4SimHits+process.materialBudgetVolumeAnalysis)
