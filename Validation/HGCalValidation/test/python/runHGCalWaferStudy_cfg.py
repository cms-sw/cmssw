###############################################################################
# Way to use this:
#   cmsRun runHGCalWaferStudy_cfg.py geometry=D110
#
#   Options for geometry D98, D99, D103, D104, D105, D106, D107, D108, D109
#                        D110, D111, D112, D113, D114, D115
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

############################################################
### SETUP OPTIONS
options = VarParsing.VarParsing('standard')
options.register('geometry',
                 "D110",
                  VarParsing.VarParsing.multiplicity.singleton,
                  VarParsing.VarParsing.varType.string,
                  "geometry of operations: D98, D99, D103, D104, D105, D106, D107, D108, D109, D110, D111, D112, D113, D114, D115")

### get and parse the command line arguments
options.parseArguments()

print(options)

############################################################
# Use the options

geomName = "Run4" + options.geometry
geomFile = "Configuration.Geometry.GeometryExtended" + geomName + "Reco_cff"
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Geometry Name:  ", geomName)
print("Geom file Name: ", geomFile)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process = cms.Process('WaferStudy',ERA)

fileInput = "file:step2" + options.geometry + "tt.root"
fileName = "hgcWafer" + options.geometry + "tt.root"

print("Input file:    ", fileInput)
print("Output file:   ", fileName)

process.load(geomFile)
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Validation.HGCalValidation.hgcalWaferStudy_cfi')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, GLOBAL_TAG, '')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.HGCalValidation=dict()

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fileInput)
                            )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService = cms.Service("TFileService",
                                   fileName = cms.string(fileName),
                                   closeFileFast = cms.untracked.bool(True)
                                   )

process.raw2digi_step = cms.Path(process.RawToDigi)
process.analysis_step = cms.Path(process.hgcalWaferStudy)
process.hgcalWaferStudy.verbosity = 1
process.hgcalWaferStudy.nBinHit   = 60
process.hgcalWaferStudy.nBinDig   = 60
process.hgcalWaferStudy.layerMinSim = cms.untracked.vint32(1,1)
process.hgcalWaferStudy.layerMaxSim = cms.untracked.vint32(10,10)
process.hgcalWaferStudy.layerMinDig = cms.untracked.vint32(1,1)
process.hgcalWaferStudy.layerMaxDig = cms.untracked.vint32(10,10)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step,process.analysis_step)
