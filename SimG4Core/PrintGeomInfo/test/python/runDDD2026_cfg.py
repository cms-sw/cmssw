###############################################################################
# Way to use this:
#   cmsRun runDDD2026_cfg.py geometry=D88
#
#   Options for geometry D77, D83, D88, D92, D93
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
                  "geometry of operations: D77, D83, D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

if (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
    materialFileName = 'matfileD83DDD.txt'
    solidFileName    = 'solidfileD83DDD.txt'
    lvFileName       = 'lvfileD83DDD.txt'
    pvFileName       = 'pvfileD83DDD.txt'
    touchFileName    = 'touchfileD83DDD.txt'
    regionFileName   = 'regionfileD83DDD.txt'
elif (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('G4PrintGeometry',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
    materialFileName = 'matfileD77DDD.txt'
    solidFileName    = 'solidfileD77DDD.txt'
    lvFileName       = 'lvfileD77DDD.txt'
    pvFileName       = 'pvfileD77DDD.txt'
    touchFileName    = 'touchfileD77DDD.txt'
    regionFileName   = 'regionfileD77DDD.txt'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('G4PrintGeometry',Phase2C11)
    process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
    materialFileName = 'matfileD92DDD.txt'
    solidFileName    = 'solidfileD92DDD.txt'
    lvFileName       = 'lvfileD92DDD.txt'
    pvFileName       = 'pvfileD92DDD.txt'
    touchFileName    = 'touchfileD92DDD.txt'
    regionFileName   = 'regionfileD92DDD.txt'
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D86Reco_cff')
    materialFileName = 'matfileD93DDD.txt'
    solidFileName    = 'solidfileD93DDD.txt'
    lvFileName       = 'lvfileD93DDD.txt'
    pvFileName       = 'pvfileD93DDD.txt'
    touchFileName    = 'touchfileD93DDD.txt'
    regionFileName   = 'regionfileD93DDD.txt'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9)
    process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')
    materialFileName = 'matfileD88DDD.txt'
    solidFileName    = 'solidfileD88DDD.txt'
    lvFileName       = 'lvfileD88DDD.txt'
    pvFileName       = 'pvfileD88DDD.txt'
    touchFileName    = 'touchfileD88DDD.txt'
    regionFileName   = 'regionfileD88DDD.txt'

print("Material file Name: ", materialFileName)
print("Solid file Name: ", solidFileName)
print("LV file Name: ", lvFileName)
print("PV file Name: ", pvFileName)
print("Touch file Name: ", touchFileName)
print("Region file Name: ", regionFileName)

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
