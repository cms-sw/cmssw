###############################################################################
# Way to use this:
#   cmsRun runDD4hep2026_cfg.py geometry=D92
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

from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
if (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9,dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D83Reco_cff')
    materialFileName = 'matfileD83DD4hep.txt'
    solidFileName    = 'solidfileD83DD4hep.txt'
    lvFileName       = 'lvfileD83DD4hep.txt'
    pvFileName       = 'pvfileD83DD4hep.txt'
    touchFileName    = 'touchfileD83DD4hep.txt'
elif (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    process = cms.Process('G4PrintGeometry',Phase2C11,dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D77Reco_cff')
    materialFileName = 'matfileD77DD4hep.txt'
    solidFileName    = 'solidfileD77DD4hep.txt'
    lvFileName       = 'lvfileD77DD4hep.txt'
    pvFileName       = 'pvfileD77DD4hep.txt'
    touchFileName    = 'touchfileD77DD4hep.txt'
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9,dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D92Reco_cff')
    materialFileName = 'matfileD92DD4hep.txt'
    solidFileName    = 'solidfileD92DD4hep.txt'
    lvFileName       = 'lvfileD92DD4hep.txt'
    pvFileName       = 'pvfileD92DD4hep.txt'
    touchFileName    = 'touchfileD92DD4hep.txt'
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9,dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D86Reco_cff')
    materialFileName = 'matfileD93DD4hep.txt'
    solidFileName    = 'solidfileD93DD4hep.txt'
    lvFileName       = 'lvfileD93DD4hep.txt'
    pvFileName       = 'pvfileD93DD4hep.txt'
    touchFileName    = 'touchfileD93DD4hep.txt'
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('G4PrintGeometry',Phase2C11M9,dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2026D88Reco_cff')
    materialFileName = 'matfileD88DD4hep.txt'
    solidFileName    = 'solidfileD88DD4hep.txt'
    lvFileName       = 'lvfileD88DD4hep.txt'
    pvFileName       = 'pvfileD88DD4hep.txt'
    touchFileName    = 'touchfileD88DD4hep.txt'

print("Material file Name: ", materialFileName)
print("Solid file Name: ", solidFileName)
print("LV file Name: ", lvFileName)
print("PV file Name: ", pvFileName)
print("Touch file Name: ", touchFileName)

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
    DD4hep           = cms.untracked.bool(False),
    Name             = cms.untracked.string(''),
    Names            = cms.untracked.vstring(''),
    MaterialFileName = cms.untracked.string(materialFileName),
    SolidFileName    = cms.untracked.string(solidFileName),
    LVFileName       = cms.untracked.string(lvFileName),
    PVFileName       = cms.untracked.string(pvFileName),
    TouchFileName    = cms.untracked.string(touchFileName),
    FileDetail       = cms.untracked.bool(True),
    type             = cms.string('PrintGeomInfoAction')
))
