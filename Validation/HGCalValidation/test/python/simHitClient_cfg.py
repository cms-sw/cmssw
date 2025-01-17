###############################################################################
# Way to use this:
#   cmsRun simHitClient_cfg.py geometry=D110
#
#   Options for geometry D98, D99, D103, D104, D105, D106, D107, D108, D109
#                        D110, D111, D112, D113, D114, D115
#
###############################################################################
import FWCore.ParameterSet.Config as cms
import os, sys, importlib, re
import FWCore.ParameterSet.VarParsing as VarParsing

####################################################################
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

####################################################################
# Use the options

geomName = "Run4" + options.geometry
print("Geometry Name:   ", geomName)
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
GLOBAL_TAG, ERA = _settings.get_era_and_conditions(geomName)
print("Global Tag Name: ", GLOBAL_TAG)
print("Era Name:        ", ERA)

process = cms.Process('SimHitClient',ERA)

geomFile = "Configuration.Geometry.GeometryExtendedRun4" + options.geometry + "Reco_cff"
fileInput = "file:SimHitVal" + options.geometry + ".root"

print("Geometry file: ", geomFile)
print("Input file:    ", fileInput)

process.load("Configuration.StandardSequences.Reconstruction_cff") 
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, GLOBAL_TAG, '')

process.load("Validation.HGCalValidation.HGCalSimHitsClient_cff")

process.load("DQMServices.Core.DQM_cfg")

# summary
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ## 

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(fileInput)
                            )

process.load("Configuration.StandardSequences.EDMtoMEAtRunEnd_cff")
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HGCalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.p = cms.Path(process.EDMtoME *
                     process.hgcalSimHitClientEE *
                     process.hgcalSimHitClientHEF *
                     process.hgcalSimHitClientHEB *
                     process.dqmSaver)
