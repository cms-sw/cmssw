###############################################################################
# Way to use this:
#   cmsRun recHitClient_cfg.py geometry=D110
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

if (options.geometry == "D115"):
    from Configuration.Eras.Era_Phase2C20I13M9_cff import Phase2C20I13M9
    process = cms.Process('Client',Phase2C20I13M9)
elif (options.geometry == "D104"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
elif (options.geometry == "D106"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
elif (options.geometry == "D109"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
elif (options.geometry == "D111"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
elif (options.geometry == "D112"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
elif (options.geometry == "D113"):
    from Configuration.Eras.Era_Phase2C22I13M9_cff import Phase2C22I13M9
    process = cms.Process('Client',PhaseC22I13M9)
else:
    from Configuration.Eras.Era_Phase2C17I13M9_cff import Phase2C17I13M9
    process = cms.Process('Client',Phase2C17I13M9)

geomFile = "Configuration.Geometry.GeometryExtendedRun4" + options.geometry + "Reco_cff"
fileInput = "file:RecHitVal" + options.geometry + ".root"

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

process.load("Validation.HGCalValidation.HGCalRecHitsClient_cff")
process.hgcalRecHitClientEE.Verbosity     = 2
process.hgcalRecHitClientHEF.Verbosity    = 2
process.hgcalRecHitClientHEB.Verbosity    = 2

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

process.load("Validation.HGCalValidation.HGCalRecHitsClient_cfi")

process.p = cms.Path(process.EDMtoME *
                     process.hgcalRecHitClientEE *
                     process.hgcalRecHitClientHEF *
                     process.hgcalRecHitClientHEB *
                     process.dqmSaver)
