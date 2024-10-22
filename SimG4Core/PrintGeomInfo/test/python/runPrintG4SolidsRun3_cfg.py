###############################################################################
# Way to use this:
#   cmsRun runPrintG4SolidsRun3_cfg.py dd4hep=False geometry=2023
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
options.register('dd4hep',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Geometry source DD4hep or DDD: False, True")
options.register('geometry',
                 "2024",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: 2021, 2023, 2024")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
if (options.dd4hep):
    from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
    process = cms.Process('PrintG4Solids',Run3_dd4hep)
    geomFile = "Configuration.Geometry.GeometryDD4hepExtended" + options.geometry + "Reco_cff"
    process.load(geomFile)
else:
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('PrintG4Solids',Run3_DDD)
    geomFile = "Configuration.Geometry.GeometryExtended" + options.geometry + "Reco_cff"
    process.load(geomFile)

process.load('FWCore.MessageService.MessageLogger_cfi')
print("Geometry file Name: ", geomFile)

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

from SimG4Core.PrintGeomInfo.g4PrintG4Solids_cfi import *

process = printGeomInfo(process)
