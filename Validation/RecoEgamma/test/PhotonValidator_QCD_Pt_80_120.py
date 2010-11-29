
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
process.GlobalTag.globaltag = 'START39_V6::All'


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

photonValidation.OutputFileName = 'PhotonValidationRelVal394_QCD_Pt_80_120.root'

photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

tkConversionValidation.OutputFileName = 'ConversionValidationRelVal394_QCD_Pt_80_120.root'
conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V6-v1/0002/5A10D0AC-39F8-DF11-814C-003048679180.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V6-v1/0001/60111483-22F8-DF11-BFC8-003048679162.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V6-v1/0001/3AC63D0A-1FF8-DF11-AF4D-002618FDA28E.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V6-v1/0001/2CDCF0A5-1DF8-DF11-9E83-0018F3D095EE.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V6-v1/0000/F2ABC8CD-ECF7-DF11-A4BB-001A92810AB8.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V6-v1/0000/D0D8B188-F7F7-DF11-AD6A-00261894394D.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V6-v1/0000/4C5C8C6F-F0F7-DF11-9E7F-001A92810AB8.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V6-v1/0000/3C71CEEB-FAF7-DF11-A617-0018F3D09600.root'

     ),
    
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/F8832A08-22F8-DF11-A243-0030486792AC.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/F052DD7F-21F8-DF11-9837-0026189438E3.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/D87CB23F-34F8-DF11-A1D9-001A9281174C.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/9C904702-23F8-DF11-A4B4-003048D15E24.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/6C304C10-1EF8-DF11-B421-003048678B8E.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/66D941A5-1DF8-DF11-8BEB-001A92810AE4.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/5295D50F-1DF8-DF11-937D-00304867BFC6.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0001/0496C890-1EF8-DF11-90EB-002618943849.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/CAA4406D-EBF7-DF11-B7E4-0026189438C2.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/A4EE06EC-F0F7-DF11-BA27-001A92811728.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/9C18A67E-F2F7-DF11-900A-0018F3D0960A.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/82D6097B-F3F7-DF11-A58A-003048678AFA.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/7CB3C06B-EDF7-DF11-9225-001A92810AA0.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/569CF366-EDF7-DF11-A10D-0026189438CF.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/14A5E4E7-FAF7-DF11-8AAD-0018F3D096EA.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/10C254EF-F6F7-DF11-B526-0030486792AC.root',
        '/store/relval/CMSSW_3_9_4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V6-v1/0000/10AF7981-F8F7-DF11-BC94-00261894391F.root'
        
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


