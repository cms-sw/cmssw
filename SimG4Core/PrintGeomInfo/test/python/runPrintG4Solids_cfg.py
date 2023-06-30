###############################################################################
# Way to use this:
#   cmsRun grunPrintG4Solids_cfg.py geometry=D88 dd4hep=False
#
#   Options for geometry D88, D91, D92, D93, D94, D95, D96, D98, D99, D100,
#                        D101
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D88",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D88, D91, D92, D93, D94, D95, D96, D98, D99, D100, D101")
options.register('dd4hep',
                 False,
                 VarParsing.VarParsing.multiplicity.singleton,
                 VarParsing.VarParsing.varType.bool,
                 "Geometry source DD4hep or DDD: False, True")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep

if (options.dd4hep):
    geomFile = "Configuration.Geometry.GeometryDD4hepExtended2026" + options.geometry + "Reco_cff"
    if (options.geometry == "D94"):
        from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
        process = cms.Process('PrintG4Solids',Phase2C20I13M9,dd4hep)
    else:
        from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
        process = cms.Process('PrintG4Solids',Phase2C17I13M9,dd4hep)
else:
    process = cms.Process('PrintG4Solids',Phase2C17I13M9)
 + "Reco_cff"
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

process = printGeomInfo(process)
