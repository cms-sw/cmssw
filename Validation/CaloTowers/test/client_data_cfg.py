import FWCore.ParameterSet.Config as cms
from DQMServices.Core.DQMEDHarvester import DQMEDHarvester

import os
import sys
import re

class config: pass
config.runNumber = int(sys.argv[1])
print config.runNumber

for arg in sys.argv: 
   print arg

readFiles = cms.untracked.vstring()

matchRootFile = re.compile("\S*\.root$")
for argument in sys.argv[2:]:
   if matchRootFile.search(argument):
      fileToRead = "file:"+argument
      readFiles.append(fileToRead)

print "readFiles : \n", readFiles

print config.runNumber

##########

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load("DQMServices.Core.DQM_cfg")
process.load("DQMServices.Components.DQMEnvironment_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag = autoCond['com10']
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run1_data', '')

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

# summary
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

#process.source = cms.Source("PoolSource",
#    fileNames = readFiles,
#    processingMode = cms.untracked.string('RunsAndLumis')
#)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = readFiles,
    filterOnRun = cms.untracked.uint32(config.runNumber)                        
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

#process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
##process.EDMtoMEConverter.convertOnEndLumi = False

cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow

#process.dqmSaver.saveByRun         = 1
#process.dqmSaver.saveAtJobEnd = False
#process.dqmSaver.forceRunNumber = 208339
#process.dqmSaver.runIsComplete = True
#process.dqmSaver.saveByRun = cms.untracked.int32(1)
#process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
#process.dqmSaver.forceRunNumber = cms.untracked.int32(999999)
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = config.runNumber

process.calotowersClient = DQMEDHarvester("CaloTowersClient", 
     outputFile = cms.untracked.string('CaloTowersHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)

process.noiseratesClient = DQMEDHarvester("NoiseRatesClient", 
     outputFile = cms.untracked.string('NoiseRatesHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)

process.hcalrechitsClient = DQMEDHarvester("HcalRecHitsClient", 
     outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
     DQMDirName = cms.string("/") # root directory
)

##########
process.calotowersDQMClient = DQMEDHarvester("CaloTowersDQMClient",
      outputFile = cms.untracked.string('CaloTowersHarvestingME.root'),
#     outputFile = cms.untracked.string(''),
      DQMDirName = cms.string("/") # root directory
)
process.hcalNoiseRatesClient = DQMEDHarvester("HcalNoiseRatesClient", 
     outputFile = cms.untracked.string('NoiseRatesHarvestingME.root'),
#     outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)
process.hcalRecHitsDQMClient = DQMEDHarvester("HcalRecHitsDQMClient", 
     outputFile = cms.untracked.string('HcalRecHitsHarvestingME.root'),
#    outputFile = cms.untracked.string(''),
     DQMDirName = cms.string("/") # root directory
)

##########

#process.p = cms.Path(process.EDMtoME * process.calotowersClient * process.noiseratesClient * process.hcalrechitsClient * process.dqmSaver)
#process.p = cms.Path(process.EDMtoME * process.calotowersClient * process.noiseratesClient * process.hcalrechitsClient * process.dqmSaver)
#process.p = cms.Path(process.EDMtoME * process.dqmSaver)
#process.p = cms.Path(process.EDMtoME * process.calotowersClient * process.dqmSaver)

process.edmtome_step = cms.Path(process.EDMtoME)
process.validationHarvesting = cms.Path(process.calotowersClient + process.noiseratesClient + process.hcalrechitsClient + process.calotowersDQMClient + process.hcalNoiseRatesClient + process.hcalRecHitsDQMClient)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition
process.schedule = cms.Schedule(process.edmtome_step,process.validationHarvesting,process.dqmsave_step)
