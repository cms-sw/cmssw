###############################################################################
# Way to use this:
#   cmsRun runDDD2026_cfg.py type=Tracker
#
#   Options for type Tracker, Calo, MTD, Muon
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('type',
                 "Tracker",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: Tracker, Calo, MTD, Muon")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('G4PrintGeometry',Phase2C17I13M9)

geomFile = "Geometry.CMSCommonData.GeometryExtended2026D98" + options.type + "Reco_cff"
materialFileName = "matfileD98" + options.type + "DDD.txt"
solidFileName    = "solidfileD98" + options.type + "DDD.txt"
lvFileName       = "lvfileD98" + options.type + "DDD.txt"
pvFileName       = "pvfileD98" + options.type + "DDD.txt"
touchFileName    = "touchfileD98" + options.type + "DDD.txt"
regionFileName   = "regionfileD98" + options.type + "DDD.txt"

print("Geometry file Name: ", geomFile)
print("Material file Name: ", materialFileName)
print("Solid file Name:    ", solidFileName)
print("LV file Name:       ", lvFileName)
print("PV file Name:       ", pvFileName)
print("Touch file Name:    ", touchFileName)
print("Region file Name:   ", regionFileName)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

from SimG4Core.PrintGeomInfo.g4PrintGeomInfo_cfi import *

process = printGeomInfo(process)

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    DumpSummary      = cms.untracked.bool(True),
    DumpLVTree       = cms.untracked.bool(False),
    DumpMaterial     = cms.untracked.bool(False),
    DumpLVList       = cms.untracked.bool(False),
    DumpLV           = cms.untracked.bool(False),
    DumpSolid        = cms.untracked.bool(True),
    DumpAttributes   = cms.untracked.bool(False),
    DumpPV           = cms.untracked.bool(False),
    DumpRotation     = cms.untracked.bool(False),
    DumpReplica      = cms.untracked.bool(False),
    DumpTouch        = cms.untracked.bool(False),
    DumpSense        = cms.untracked.bool(False),
    DumpRegion       = cms.untracked.bool(False),
    DD4hep           = cms.untracked.bool(False),
    Name             = cms.untracked.string(''),
    Names            = cms.untracked.vstring(''),
    MaterialFileName = cms.untracked.string(materialFileName),
    SolidFileName    = cms.untracked.string(solidFileName),
    LVFileName       = cms.untracked.string(lvFileName),
    PVFileName       = cms.untracked.string(pvFileName),
    TouchFileName    = cms.untracked.string(touchFileName),
    RegionFileName   = cms.untracked.string(regionFileName),
    FileDetail       = cms.untracked.bool(True),
    type             = cms.string('PrintGeomInfoAction')
))

if (options.type == "Tracker"):
    process.g4SimHits.OnlySDs = ['TkAccumulatingSensitiveDetector']
elif (options.type == "MTD"):
    process.g4SimHits.OnlySDs = ['MtdSensitiveDetector']
elif (options.type == "Muon"):
    process.g4SimHits.OnlySDs = ['MuonSensitiveDetector']
else:
    process.g4SimHits.OnlySDs = ['CaloTrkProcessing', 'EcalSensitiveDetector', 'HcalSensitiveDetector', 'HGCalSensitiveDetector', 'HFNoseSensitiveDetector', 'HGCScintillatorSensitiveDetector', 'ZdcSensitiveDetector']
