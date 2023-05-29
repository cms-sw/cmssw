###############################################################################
# Way to use this:
#   cmsRun runPrintG4SolidsRun3_cfg.py dd4hep=False
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, imp, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
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
if (options.dd4hep):
    from Configuration.Eras.Era_Run3_dd4hep_cff import Run3_dd4hep
    process = cms.Process('PrintG4Solids',Run3_dd4hep)
    process.load('Configuration.Geometry.GeometryDD4hepExtended2021Reco_cff')
else:
    from Configuration.Eras.Era_Run3_DDD_cff import Run3_DDD
    process = cms.Process('PrintG4Solids',Run3_DDD)
    process.load('Configuration.Geometry.GeometryExtended2021Reco_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

from SimG4Core.PrintGeomInfo.g4PrintG4Solids_cfi import *

process = printGeomInfo(process)
