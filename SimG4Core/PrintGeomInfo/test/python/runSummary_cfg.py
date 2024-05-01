###############################################################################
# Way to use this:
#   cmsRun runSummary_cfg.py geometry=2023
#
#   Options for geometry 2021, 2023, 2024
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "2024",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2021, 2023, 2024")

### get and parse the command line arguments
options.parseArguments()

print(options)

#####p###############################################################
# Use the options

from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
process = cms.Process('PrintGeometry',Run3_DDD)
geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"

print("Geometry file: ", geomFile)

process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cout=dict()
    process.MessageLogger.G4cerr=dict()

from SimG4Core.PrintGeomInfo.g4PrintGeomSummary_cfi import *

process = printGeomSummary(process)
