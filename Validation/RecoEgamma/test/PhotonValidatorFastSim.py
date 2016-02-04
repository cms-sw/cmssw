import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")
process.load('Configuration/StandardSequences/GeometryPilot2_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidationSequence_cff")
process.load("Validation.RecoEgamma.photonPostprocessing_cfi")
process.load("Validation.RecoEgamma.conversionPostprocessing_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START311_V0::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *


photonValidation.OutputFileName = 'PhotonValidationRelVal3_11_0_pre5_H130GGgluonfusion_FastSim_2.root'
photonValidation.fastSim = True
photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

#conversionPostprocessing.standalone = cms.bool(True)
#conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
#conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0064/00C76A96-0424-E011-9AC4-0026189437F8.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/F0ABACF8-CC23-E011-9B7F-00248C55CC40.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/C2B098EC-CA23-E011-BB2D-002618943868.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/AEC5F1F1-CA23-E011-8BDD-002618943911.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/A0EEDA74-CB23-E011-90ED-00248C55CC40.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/8AF5A671-CB23-E011-891B-00261894382D.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/824715EC-CA23-E011-B387-002618FDA216.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/74DD72F2-CA23-E011-8BD5-0026189437FA.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/582A6B7B-CD23-E011-B2C1-0018F3D096E8.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/56D7057C-CD23-E011-B50B-00304867D836.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/50B4A078-CD23-E011-BE5E-00248C55CC40.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/2A52ABEE-CA23-E011-8481-001A92811708.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/10479EF3-CA23-E011-ACEC-001A928116DE.root',
        '/store/relval/CMSSW_3_11_0_pre5/RelValH130GGgluonfusion/GEN-SIM-DIGI-RECO/START311_V0_FastSim-v1/0063/0AE6F7F1-CA23-E011-95A4-0026189437FA.root'
    )
    )


photonPostprocessing.rBin = 48

## For single gamma pt =10
#photonValidation.eMax  = 100
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonPostprocessing.eMax  = 100
#photonPostprocessing.etMax = 50

## For single gamma pt = 35
#photonValidation.eMax  = 300
#photonValidation.etMax = 50
#photonValidation.etScale = 0.20
#photonValidation.dCotCutOn = False
#photonValidation.dCotCutValue = 0.15


## For gam Jet and higgs
photonValidation.eMax  = 500
#photonValidation.etMax = 500
photonValidation.etMax = 250
photonPostprocessing.eMax  = 500
photonPostprocessing.etMax = 250



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)




#process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.p1 = cms.Path(process.tpSelection*process.photonValidation*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
