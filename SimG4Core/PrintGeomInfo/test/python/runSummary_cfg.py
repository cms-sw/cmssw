###############################################################################
# Way to use this:
#   cmsRun runSummary_cfg.py geometry=Run3
#
#   Options for geometry Run3, D88, D92, D93
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "Run3",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: Run3, D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

#####p###############################################################
# Use the options

if (options.geometry == "Run3"):
    geomFile = "Configuration.Geometry.GeometryExtended2021Reco_cff"
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('PrintGeometry',Run3_DDD)
else:
    geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    process = cms.Process('PrintGeometry',Phase2C11M9)

print("Geometry file: ", geomFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()

from SimG4Core.PrintGeomInfo.g4PrintGeomSummary_cfi import *

process = printGeomSummary(process)
