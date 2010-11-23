
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
process.load("Validation.RecoEgamma.tkConvValidator_cfi")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START39_V3::All'


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

photonValidation.OutputFileName = 'PhotonValidationRelVal392_QCD_Pt_80_120.root'

photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

tkConversionValidation.OutputFileName = 'ConversionValidationRelVal392_QCD_Pt_80_120.root'
conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0072/30E8A6A8-9AE9-DF11-AFB3-002618943900.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0070/ACC7591F-35E8-DF11-A970-001A928116C0.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0070/8412F11E-35E8-DF11-8A7F-0018F3D096EE.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0070/7C32791A-35E8-DF11-8056-002618943963.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0068/DE28BD57-24E8-DF11-BA76-001A92810AC8.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0068/B8469055-24E8-DF11-AD11-0018F3D09654.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0068/B09CEC5A-24E8-DF11-89E4-001A92971BD6.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V3-v1/0068/9667C6BE-1CE8-DF11-BF92-001A928116F4.root'



     ),
    
    secondaryFileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0073/D28A9828-A8E9-DF11-86C5-002618943911.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0070/F2078161-3BE8-DF11-A302-0018F3D09608.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0070/EE4D965A-3BE8-DF11-9EC6-0026189438EF.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0070/AAAC445F-3BE8-DF11-BEDA-001A928116AE.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0070/8C74595F-3BE8-DF11-910D-0018F3D09616.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0070/829FD661-3BE8-DF11-AC4B-003048678AC0.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0070/7C95D05E-3BE8-DF11-904B-001A92971B84.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/C2791B50-25E8-DF11-8E90-00304867C0F6.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/BE353A47-25E8-DF11-A113-001A92810AA0.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/BA2BF44A-25E8-DF11-97BD-00261894395B.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/5824034B-25E8-DF11-A470-0018F3D09688.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/52D0D427-1CE8-DF11-B2C9-003048679080.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/4E5C2EB9-1CE8-DF11-8FA8-003048678A88.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/3602A93E-25E8-DF11-A11C-00261894387B.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/324AEE46-25E8-DF11-9422-002618943945.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/1A88D4BB-1CE8-DF11-BC2C-001A928116EE.root',
        '/store/relval/CMSSW_3_9_2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0068/084D3F40-25E8-DF11-820F-0018F3D0962C.root'


        
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

process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)


