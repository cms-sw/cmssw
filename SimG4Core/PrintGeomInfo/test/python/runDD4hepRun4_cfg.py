###############################################################################
# Way to use this:
#   cmsRun runDD4hepRun4_cfg.py geometry=D110
#
#   Options for geometry D95, D96, D98, D99, D100, D101, D102, D103, D104,
#                        D105, D106, D107, D108, D109, D110, D111, D112, D113,
#                        D114, D115
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D110",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D95, D96, D98, D99, D100, D101, D102, D103, D104, D105, D106, D107, D108, D109, D110, D111, D112, D113, D114, D115")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

geomName = "Run4" + options.geometry
geomFile = "Configuration.Geometry.GeometryDD4hepExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry Name:   ", geomName)
print("Geom file Name:  ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
process = cms.Process('G4PrintGeometry',ERA,dd4hep)

materialFileName = "matfile" + options.geometry + "DD4hep.txt"
solidFileName    = "solidfile" + options.geometry + "DD4hep.txt"
lvFileName       = "lvfile" + options.geometry + "DD4hep.txt"
pvFileName       = "pvfile" + options.geometry + "DD4hep.txt"
touchFileName    = "touchfile" + options.geometry + "DD4hep.txt"
regionFileName   = "regionfile" + options.geometry + "DD4hep.txt"

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
