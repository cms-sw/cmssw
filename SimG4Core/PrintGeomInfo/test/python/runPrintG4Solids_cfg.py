###############################################################################
# Way to use this:
#   cmsRun grunPrintG4Solids_cfg.py geometry=D98 dd4hep=False
#
#   Options for geometry D88, D91, D92, D93, D94, D95, D96, D98, D99, D100,
#                        D101
#   Options for type DDD, DD4hep
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
                  "geometry of operations: D88, D91, D92, D93, D94, D95, D96, D98, D99, D100, D101")
options.register('type',
                 "DDD",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "type of operations: DDD, DD4hep")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

if (options.type == "DD4hep"):
    geomFile = "Configuration.Geometry.GeometryDD4hepExtended2026" + options.geometry + "Reco_cff"
    if (options.geometry == "D94"):
        from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
        process = cms.Process('PrintG4Solids',Phase2C20I13M9,dd4hep)
    else:
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('PrintG4Solids',Phase2C17I13M9,dd4hep)
else:
    geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
    if (options.geometry == "D94"):
        from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
        process = cms.Process('PrintG4Solids',Phase2C20I13M9)
    else:
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('PrintG4Solids',Phase2C17I13M9)

print("Geometry file Name: ", geomFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

from SimG4Core.PrintGeomInfo.g4PrintG4Solids_cfi import *

if (options.type == "DD4hep"):
    process.g4SimHits.Watchers.dd4hep = True

process = printGeomInfo(process)
