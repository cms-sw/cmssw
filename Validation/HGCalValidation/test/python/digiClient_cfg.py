###############################################################################
# Way to use this:
#   cmsRun digiClient_cfg.py geometry=D92
#
#   Options for geometry D88, D92, D93
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
                  "geometry of operations: D88, D92, D93")

### get and parse the command line arguments
options.parseArguments()

print(options)

####################################################################
# Use the options

from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
process = cms.Process('Client',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtended2026" + options.geometry + "Reco_cff"
fileInput = "file:DigiVal" + options.geometry + ".root"

print("Geometry file: ", geomFile)
print("Input file:    ", fileInput)

process.load("Configuration.StandardSequences.Reconstruction_cff") 
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load(geomFile)
process.load('FWCore.MessageService.MessageLogger_cfi')
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T21', '')

process.load("Validation.HGCalValidation.HGCalDigiClient_cff")
process.hgcalDigiClientEE.Verbosity     = 2
process.hgcalDigiClientHEF.Verbosity    = 2
process.hgcalDigiClientHEB.Verbosity    = 2

process.load("DQMServices.Core.DQM_cfg")

# summary
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) ) ##

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
    )

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring( fileInput )
                            )

process.load("Configuration.StandardSequences.EDMtoMEAtRunEnd_cff")
process.dqmSaver.referenceHandling = cms.untracked.string('all')

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HGCalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

process.load("Validation.HGCalValidation.HGCalDigiClient_cfi")

process.p = cms.Path(process.EDMtoME *
                     process.hgcalDigiClientEE *
                     process.hgcalDigiClientHEF *
                     process.hgcalDigiClientHEB *
                     process.dqmSaver)
