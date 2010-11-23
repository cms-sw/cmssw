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

photonValidation.OutputFileName = 'PhotonValidationRelVal392_PhotonJets_Pt_10.root'

photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName



process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0073/88CE922F-A8E9-DF11-9E64-0018F3D09704.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0067/F258556E-0EE8-DF11-915E-00248C55CC9D.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0067/DC3643DD-0CE8-DF11-B061-0026189437EB.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0067/D6B36F60-13E8-DF11-BDDA-0018F3D096BA.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-RECO/START39_V3-v1/0067/5E883FE1-10E8-DF11-AA9F-0026189438DA.root'


    ),


    secondaryFileNames = cms.untracked.vstring(
	'/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0072/2A29A05F-96E9-DF11-B32F-002618943915.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/F0F570DA-0EE8-DF11-9C43-001A92971B08.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/DAED27DD-13E8-DF11-90C5-00261894391F.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/A8C1EDDC-10E8-DF11-9EC0-003048678BC6.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/62455864-13E8-DF11-9E87-001A92971B94.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/46E0D0C8-0CE8-DF11-BB26-001A92971B72.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/3671D0BB-0CE8-DF11-9D4A-003048678D86.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/30EF7151-12E8-DF11-B319-00304867916E.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/2ACB2BBD-0CE8-DF11-9A72-00261894383E.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/2831AA39-0DE8-DF11-9C34-00261894395B.root',
        '/store/relval/CMSSW_3_9_2/RelValPhotonJets_Pt_10/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V3-v1/0067/184816BC-0DE8-DF11-A02A-0026189437EB.root'



    )
 )


photonPostprocessing.rBin = 48
## For gam Jet and higgs
photonValidation.eMax  = 100
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonPostprocessing.eMax  = 100
photonPostprocessing.etMax = 50




process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)

process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
