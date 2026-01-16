###############################################################################
# Way to use this:
#   cmsRun runSummaryRun4_cfg.py geometry=D121
#
#   Options for geometry D104, D110, D111, D112, D113, D114, D115, D120, D121,
#                        D122, D123, D124, D125
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D121",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D104, D110, D111, D112, D113, D114, D115, D120, D121, D122, D123, D124, D125")

### get and parse the command line arguments
options.parseArguments()

print(options)

#####p###############################################################
# Use the options

geomName = "Run4" + options.geometry
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry Name:   ", geomName)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process = cms.Process('G4PrintGeometry',ERA)
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"

print("Geometry file Name: ", geomFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()

from SimG4Core.PrintGeomInfo.g4PrintGeomSummary_cfi import *

process = printGeomSummary(process)
