###############################################################################
# Way to use this:
#   cmsRun runDD4hep2026_cfg.py geometry=D92
#
#   Options for geometry D86, D88, D91, D92, D93, D94, D95, D96, D97, D98, D99,
#                        D100, D101
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D92",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D86, D88, D91, D92, D93, D94, D95, D96, D97, D98, D99, D100, D101")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
if (options.geometry == "D94"):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    process = cms.Process('G4PrintGeometry',Phase2C20I13M9,dd4hep)
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9,dd4hep)

geomFile = "Configuration.Geometry.GeometryDD4hepExtended2026" + options.geometry + "Reco_cff"
materialFileName = "matfile" + options.geometry + "DD4hep.txt"
solidFileName    = "solidfile" + options.geometry + "DD4hep.txt"
lvFileName       = "lvfile" + options.geometry + "DD4hep.txt"
pvFileName       = "pvfile" + options.geometry + "DD4hep.txt"
touchFileName    = "touchfile" + options.geometry + "DD4hep.txt"
regionFileName   = "regionfile" + options.geometry + "DD4hep.txt"

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
