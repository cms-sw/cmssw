import os
import FWCore.ParameterSet.Config as cms


process = cms.Process("DigiValidation")
process.load("Configuration.Geometry.GeometryHCAL_cff")
process.load("Configuration.StandardSequences.Simulation_cff")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration/StandardSequences/DigiToRaw_cff')
process.load('Configuration/StandardSequences/RawToDigi_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['mc']


process.load("DQMServices.Core.DQM_cfg")
process.DQM.collectorHost = ''

process.options = cms.untracked.PSet(
                                     Rethrow=cms.untracked.vstring('ProductNotFound')
                                     )

### to official ###
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load('Configuration/StandardSequences/EDMtoMEAtRunEnd_cff')
process.dqmSaver.referenceHandling = cms.untracked.string('all')
cmssw_version = os.environ.get('CMSSW_VERSION','CMSSW_X_Y_Z')
Workflow = '/HcalValidation/'+'Harvesting/'+str(cmssw_version)
process.dqmSaver.workflow = Workflow


process.maxEvents = cms.untracked.PSet(
                                       input=cms.untracked.int32(10)
                                       )

process.source = cms.Source("PoolSource",
                            #    fileNames = cms.untracked.vstring("file:RAW.root")
                            fileNames=cms.untracked.vstring(

                            )
)


process.hcalDigiAnalyzer = cms.EDAnalyzer("HcalDigisValidation",
                                          outputFile=cms.untracked.string('HcalDigisValidationRelVal.root'),
                                          digiLabel=cms.InputTag("hcalDigis"),
                                          mode=cms.untracked.string('multi'),
                                          hcalselector=cms.untracked.string('all'),
                                          mc=cms.untracked.string('yes') # 'yes' for MC
                                          )

process.hcaldigisClient = cms.EDAnalyzer("HcalDigisClient",
                                         outputFile=cms.untracked.string('HcalDigisHarvestingME.root'),
                                         DQMDirName=cms.string("/") # root directory
                                         )



#--- to force RAW->Digi
#process.hcalDigis.InputLabel = 'source'             # data
process.hcalDigis.InputLabel = 'rawDataCollector'    # MC

process.p = cms.Path(process.hcalDigis * process.hcalDigiAnalyzer * process.hcaldigisClient * process.dqmSaver)



