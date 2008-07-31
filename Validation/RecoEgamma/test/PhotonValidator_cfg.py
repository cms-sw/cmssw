import FWCore.ParameterSet.Config as cms

process = cms.Process("TestPhotonValidator")

process.load("RecoEcal.EgammaClusterProducers.geometryForClustering_cff")
process.load("Configuration.StandardSequences.MagneticField_40T_cff")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Geometry.TrackerGeometryBuilder.trackerGeometry_cfi")
process.load("RecoTracker.GeometryESProducer.TrackerRecoGeometryESProducer_cfi")
process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")
process.load("RecoTracker.MeasurementDet.MeasurementTrackerESProducer_cfi")
process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")
process.load("SimTracker.TrackAssociation.TrackAssociatorByHits_cfi")
process.load("SimGeneral.MixingModule.mixNoPU_cfi")
process.load("DQMServices.Components.MEtoEDMConverter_cfi")
process.load("Validation.RecoEgamma.photonValidator_cfi")

process.DQMStore = cms.Service("DQMStore");



#  include "DQMServices/Components/data/MessageLogger.cfi"
#  service = LoadAllDictionaries {}



process.maxEvents = cms.untracked.PSet(
 # input = cms.untracked.int32(1000)
)


from Validation.RecoEgamma.photonValidator_cfi import *
<<<<<<< PhotonValidator_cfg.py
#photonValidation.OutputFileName = 'sega.root'
photonValidation.OutputFileName = 'PhotonValidationRelVal210_pre6_SingleGammaPt35WithFixedAss.root'
#photonValidation.OutputFileName = 'PhotonValidationRelVal210_pre6_H130GGgluonfusion.root'
=======
photonValidation.OutputFileName = 'PhotonValidationRelVal210_pre9_SingleGammaPt35.root'
>>>>>>> 1.3


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
<<<<<<< PhotonValidator_cfg.py



 ## Official RelVal 210pre6 Single Photon pt=35 GeV 4T   
    'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/28C3DDA9-B03E-DD11-AAA9-000423D9A212.root',
    'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/34A839BD-AF3E-DD11-ADFC-000423D99E46.root',
    'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/4ECD9899-B23E-DD11-90A9-001617C3B6E2.root',
    'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt35-1213920853-IDEAL_V2-2nd/0000/CC46175D-B03E-DD11-8EE1-000423D9890C.root')

 ## Official RelVal 210pre6 Single Photon pt=10 GeV 4T   
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt10-1213987236-IDEAL_V2-2nd/0004/101599B7-E240-DD11-AB85-001617DBD5AC.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt10-1213987236-IDEAL_V2-2nd/0004/227BFB9E-0241-DD11-8BA4-001617DBD5AC.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt10-1213987236-IDEAL_V2-2nd/0004/5A2713AC-D640-DD11-AD1A-000423D992A4.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt10-1213987236-IDEAL_V2-2nd/0004/A2D92749-0C41-DD11-A876-000423D944F8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt10-1213987236-IDEAL_V2-2nd/0005/22402E61-1B41-DD11-9051-000423D986A8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt10-1213987236-IDEAL_V2-2nd/0005/24330E47-0B41-DD11-B8A3-001D09F25109.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt10-1213987236-IDEAL_V2-2nd/0005/26BDD1DC-0741-DD11-A3A3-0019B9F72BAA.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt10-1213987236-IDEAL_V2-2nd/0005/9A72477E-0341-DD11-A884-000423D94C68.root')


## Official RelVal 210pre6 Single Photon pt=35 GeV 3.8 T
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0004/6AF10932-F340-DD11-9BB7-000423D94700.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0004/AAFFE6D9-0141-DD11-9019-001617DBCF6A.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/245E3E90-0741-DD11-BB73-000423D99660.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/2A7A33B3-1841-DD11-8002-000423D6B42C.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/4CD97038-0741-DD11-8FD7-001D09F23174.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/9058D215-0541-DD11-BA20-000423D986C4.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/22/RelVal-RelValSingleGammaPt35-1213987236-IDEAL_V2-2nd/0005/E2B6920E-0541-DD11-9144-000423D99A8E.root' )


## Official RelVal 210pre6 Single Photon pt=1000 GeV 4T   
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/10FE12B7-AF3E-DD11-9B4C-000423D6CA6E.root',
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/4A9C5FF7-AE3E-DD11-9E71-000423D992A4.root',
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/5294E693-AE3E-DD11-9E19-001617C3B710.root',
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/9EFA96BC-AF3E-DD11-88B1-001617E30D06.root',
 #'rfio:/castor/cern.ch/cms/store/relval/2008/6/20/RelVal-RelValSingleGammaPt1000-1213920853-IDEAL_V2-2nd/0000/A6FE0D26-AF3E-DD11-A4DE-001617C3B710.root')

=======
 
## Official RelVal 210pre9 Single Photon pt=10
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt10-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/4862131C-6D57-DD11-847F-000423D996C8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt10-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/E421F150-6D57-DD11-ABEC-000423DD2F34.root')

## Official RelVal 210pre9 Single Photon pt=35
'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt35-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/60AF541B-6D57-DD11-9EB6-001617C3B77C.root',
'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt35-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/BE093BF5-6C57-DD11-9258-001617C3B5F4.root')

## Official RelVal 210pre9 Single Photon pt=1000
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt1000-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/140FD6C7-6C57-DD11-8F00-000423D94534.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt1000-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/3AC9638C-6C57-DD11-AED4-001617C3B70E.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt1000-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/7C2E07C4-6C57-DD11-AB1F-000423D98B5C.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/7/21/RelVal-RelValSingleGammaPt1000-1216579481-IDEAL_V5-2nd/RelValSingleGammaPt1000/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/CMSSW_2_1_0_pre9-RelVal-1216579481-IDEAL_V5-2nd-unmerged/0000/EE1DBDFC-6C57-DD11-8FF0-000423D6B358.root')
>>>>>>> 1.3


## Official RelVal 210pre6 H130GGgluonfusion
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0006/62EAD706-A942-DD11-9B2D-000423D98EC8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0006/888867E4-A942-DD11-A96E-000423DD2F34.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0006/92F4C43B-A242-DD11-9FC3-000423D98DB4.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0006/C4C58A62-A342-DD11-B9AE-000423D6CAF2.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0006/D224B28D-A242-DD11-A14E-000423D99F3E.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/007212FC-AD42-DD11-877F-000423D6CAF2.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/0072FE5E-AA42-DD11-95D8-000423D98BE8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/00B7395D-BD42-DD11-B9E4-000423D9880C.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/06C982F0-CD42-DD11-9677-000423D6B358.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/0E67705E-AE42-DD11-B795-000423D98EC8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/1A24196B-A842-DD11-9697-001617C3B5F4.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/1EF6C1A9-BD42-DD11-983B-000423D9863C.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/22088250-B442-DD11-9DC6-001617DBD316.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/244200EC-AE42-DD11-9177-001617C3B6DC.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/266F6E21-AA42-DD11-86EC-001617C3B6E8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/28E12784-AB42-DD11-BF60-001617C3B6CE.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/34AA8493-AB42-DD11-940A-000423D98AF0.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/38F57B16-B542-DD11-A544-000423D6CA6E.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/3A7188F2-AD42-DD11-8671-001617DF785A.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/46D651A1-B542-DD11-9E26-000423D6B5C4.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/48882052-AF42-DD11-A5A7-000423D6CA42.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/5215C2C8-A542-DD11-8ABF-001617E30D40.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/54F138CB-E442-DD11-9381-000423D9853C.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/603BC7A2-AB42-DD11-B338-001617C3B710.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/60BDADA8-AB42-DD11-81E9-000423D99AA2.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/6A7783D9-B342-DD11-B2BE-000423D9939C.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/6E8CA85D-B542-DD11-AB7B-000423D98DC4.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/6EA6C3D8-B942-DD11-9298-0019DB29C5FC.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/76524DAF-AA42-DD11-AE23-000423D98920.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/7667154C-B342-DD11-967B-001617E30E28.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/76FEDF9C-A642-DD11-AD79-001617DBD5AC.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/7A01E22A-AB42-DD11-9D60-001617DBD332.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/80F7841E-B442-DD11-9428-000423D94700.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/84441CF9-AA42-DD11-B5A5-000423DD2F34.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/84C32FFF-AD42-DD11-A432-000423D99160.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/883FD762-A542-DD11-B278-000423D99AAE.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/9C13A4EE-AA42-DD11-81FE-001617E30D06.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/A4C7C071-AE42-DD11-9308-001617C3B70E.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/A8A09BFD-B542-DD11-A6FB-001617E30D4A.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/AA6B63A5-AB42-DD11-83DD-000423D99AAA.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/B0B553BA-AB42-DD11-A3BC-00161757BF42.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/B4E23F0B-AC42-DD11-B4D5-000423D94E70.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/B616D2D7-AD42-DD11-AE89-000423D6CAF2.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/B6B11920-A642-DD11-A7F9-0019DB29C614.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/BC4965CD-A542-DD11-BDEB-000423D98930.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/CAF7811A-AD42-DD11-8001-000423D996C8.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/E231657E-AA42-DD11-9C34-000423D6CA02.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/E2DA0964-AE42-DD11-A6AB-000423D9939C.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/E4702C09-AE42-DD11-87DA-000423D998BA.root',
#'rfio:/castor/cern.ch/cms/store/relval/2008/6/25/RelVal-RelValH130GGgluonfusion-1214239099-STARTUP_V1-2nd/0007/EEE4BB0B-AD42-DD11-B9E3-000423D6CAF2.root')





)



from SimTracker.TrackAssociation.TrackAssociatorByHits_cfi import *
import SimTracker.TrackAssociation.TrackAssociatorByHits_cfi
#TrackAssociatorByHits.AbsoluteNumberOfHits = True
#TrackAssociatorByHits.Cut_RecoToSim = 3
#TrackAssociatorByHits.Quality_SimToReco = 3
TrackAssociatorByHits.Cut_RecoToSim = 0.5
TrackAssociatorByHits.Quality_SimToReco = 0.5


process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.photonValidation)
process.schedule = cms.Schedule(process.p1)



