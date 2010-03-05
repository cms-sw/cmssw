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
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_3XY_V21::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *

photonValidation.OutputMEsInRootFile = True
photonValidation.OutputFileName = 'PhotonValidationRelVal352_H130GGgluonfusion.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(



# official RelVal 352 RelValH130GGgluonfusion

        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0016/F0F46C9B-3B1E-DF11-936A-001731A28F19.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0016/B6A3E54F-491E-DF11-B4B5-0018F3D09676.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0016/88DED6BF-301E-DF11-95D7-0018F3D096F8.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0016/7E554072-391E-DF11-8FA3-0017312B5DC9.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0016/709A1C5A-D91E-DF11-AEDA-003048678F74.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-RECO/START3X_V21-v1/0016/1C2F0B70-3A1E-DF11-94D9-001731AF66C1.root'

    ),
    secondaryFileNames = cms.untracked.vstring(



# official RelVal 352 RelValH130GGgluonfusion
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/F068BAD5-3A1E-DF11-A4B5-001731A281B1.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/C83D97C0-2F1E-DF11-AA4A-0018F3D0960A.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/C0C9756C-3A1E-DF11-8FFF-001731A28BE1.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/B4111D48-301E-DF11-BD44-0018F3D09706.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/AAD40BF4-311E-DF11-96EF-001A928116AE.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/A6AD696D-3A1E-DF11-8B0E-001731AF66C1.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/9C1070F3-311E-DF11-8F2F-001A92971B88.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/801DC6F4-391E-DF11-83A8-001731AF685D.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/3E015B94-3B1E-DF11-B2C4-0018F3D0970E.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/382FB368-3C1E-DF11-BAE2-001A92811714.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/22E5234C-D91E-DF11-9267-003048D3C010.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/1E3133F4-391E-DF11-A638-001731AF66A5.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/089CA866-391E-DF11-A288-0017312A250B.root',
        '/store/relval/CMSSW_3_5_2/RelValH130GGgluonfusion/GEN-SIM-DIGI-RAW-HLTDEBUG/START3X_V21-v1/0016/083A4F63-391E-DF11-8AB2-001A92810AC4.root'
    
    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
photonValidation.eMax  = 500
photonValidation.etMax = 500
photonPostprocessing.eMax  = 500
photonPostprocessing.etMax = 500




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
