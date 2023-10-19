###############################################################################
# Way to use this:
#   cmsRun runSummary2026_cfg.py geometry=D92
#
#   Options for geometry D86, D88, D91, D92, D93, D94, D95, D96, D98, D99,
#                        D100, D101
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
                  "geometry of operations: D86, D88, D91, D92, D93, D94, D95, D96, D98, D99, D100, D101")

### get and parse the command line arguments
options.parseArguments()

print(options)

#####p###############################################################
# Use the options

if (options.geometry == "D94"):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    process = cms.Process('G4PrintGeometry',Phase2C20I13M9)
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('G4PrintGeometry',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"

print("Geometry file Name: ", geomFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()

from SimG4Core.PrintGeomInfo.g4PrintGeomSummary_cfi import *

process = printGeomSummary(process)
