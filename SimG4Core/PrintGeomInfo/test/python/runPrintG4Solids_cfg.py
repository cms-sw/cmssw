###############################################################################
# Way to use this:
#   cmsRun grunPrintG4Solids_cfg.py geometry=D86 dd4hep=False
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
if (options.geometry == "D83"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    if (options.dd4hep):
        process = cms.Process('PrintG4Solids',Phase2C11M9,dd4hep)
        process.load('Configuration.Geometry.GeometryDD4hepExtended2026D83Reco_cff')
    else:
        process = cms.Process('PrintG4Solids',Phase2C11M9)
        process.load('Configuration.Geometry.GeometryExtended2026D83Reco_cff')
elif (options.geometry == "D77"):
    from Configuration.Eras.Era_Phase2C11_cff import Phase2C11
    if (options.dd4hep):
        process = cms.Process('PrintG4Solids',Phase2C11,dd4hep)
        process.load('Configuration.Geometry.GeometryDD4hepExtended2026D77Reco_cff')
    else:
        process = cms.Process('PrintG4Solids',Phase2C11)
        process.load('Configuration.Geometry.GeometryExtended2026D77Reco_cff')
elif (options.geometry == "D92"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    if (options.dd4hep):
        process = cms.Process('PrintG4Solids',Phase2C11M9,dd4hep)
        process.load('Configuration.Geometry.GeometryDD4hepExtended2026D92Reco_cff')
    else:
        process = cms.Process('PrintG4Solids',Phase2C11M9)
        process.load('Configuration.Geometry.GeometryExtended2026D92Reco_cff')
elif (options.geometry == "D93"):
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    if (options.dd4hep):
        process = cms.Process('PrintG4Solids',Phase2C11M9,dd4hep)
        process.load('Configuration.Geometry.GeometryDD4hepExtended2026D93Reco_cff')
    else:
        process = cms.Process('PrintG4Solids',Phase2C11M9)
        process.load('Configuration.Geometry.GeometryExtended2026D93Reco_cff')
else:
    from Configuration.Eras.Era_Phase2C11M9_cff import Phase2C11M9
    if (options.dd4hep):
        process = cms.Process('PrintG4Solids',Phase2C11M9,dd4hep)
        process.load('Configuration.Geometry.GeometryDD4hepExtended2026D88Reco_cff')
    else:
        process = cms.Process('PrintG4Solids',Phase2C11M9)
        process.load('Configuration.Geometry.GeometryExtended2026D88Reco_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.G4cerr=dict()
    process.MessageLogger.G4cout=dict()

from SimG4Core.PrintGeomInfo.g4PrintG4Solids_cfi import *

process = printGeomInfo(process)
