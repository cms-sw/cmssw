
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
process.GlobalTag.globaltag = 'START38_V6::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre8_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre8 QCD_Pt_80_120
  '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/EAD3F841-8D8E-DF11-8DA9-00248C55CC9D.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/BAA334DA-8C8E-DF11-84FA-003048678B44.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/AC5F74D3-8C8E-DF11-8F6D-002618943856.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/AC5CF5C8-8C8E-DF11-8F47-002618FDA287.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/A071D8E1-8C8E-DF11-AF92-0026189437FD.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/A046BBC5-8C8E-DF11-BE7B-003048678BB8.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/A023A152-8D8E-DF11-B098-003048678FFE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/96FFEC34-8D8E-DF11-A67F-0018F3D09652.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/84B1BEE1-8C8E-DF11-B697-0018F3D09650.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/822C88D1-8C8E-DF11-AB22-0018F3D09650.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/5ADA11D1-8C8E-DF11-A42B-0026189438FF.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/58AB23B3-8D8E-DF11-B759-00261894387D.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/4AB336E3-8C8E-DF11-8EC0-003048679214.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/444972F9-8C8E-DF11-BEF8-002618FDA211.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/36DE4D19-8D8E-DF11-824B-0018F3D0968E.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/26212E15-8D8E-DF11-A859-003048678B0E.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/222AB362-8D8E-DF11-8451-00261894394F.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0004/202ADF6F-8D8E-DF11-8BD8-00261894386C.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0001/1066589C-A58B-DF11-A2B5-0026189438C0.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0000/F08609DA-6F8B-DF11-9B48-001A92810ADE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0000/D4576267-6B8B-DF11-BD7D-0026189438B4.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0000/78FEE9FB-6D8B-DF11-AAFD-0018F3D09648.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0000/664B9D88-6B8B-DF11-80B9-002618943985.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0000/2666D0DB-6E8B-DF11-8B1F-0018F3D096CA.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0000/22D830ED-6B8B-DF11-898F-001A928116C6.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V6-v1/0000/1C41F9D2-6E8B-DF11-8531-001A92811732.root'
 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre8 QCD_Pt_80_120


 '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/FEEF54BB-8C8E-DF11-8824-003048678AC8.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/FAB6AEB2-8C8E-DF11-ADEA-003048678C62.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/FA8F41BA-8C8E-DF11-9C49-003048D15D04.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/EED42BA6-8C8E-DF11-9A63-003048678B0E.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/E24580BF-8C8E-DF11-826E-002618FDA216.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/DEC8B956-8D8E-DF11-9C5B-003048679164.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/D25AD3C1-8C8E-DF11-94B2-003048678B18.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/C815A0CB-8C8E-DF11-A5A0-00304867C16A.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/BE1DE0B5-8C8E-DF11-9FB1-003048678FB4.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/BE0DE547-8D8E-DF11-AB57-00304867C16A.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/BA5518F9-8C8E-DF11-9F3C-001A92971BCA.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/B6B0E641-8D8E-DF11-ABEA-003048678C62.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/B44FA3BB-8C8E-DF11-9109-0018F3D09650.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/AE38C001-8D8E-DF11-9FDA-0018F3D09652.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/A097FFBC-8C8E-DF11-BAC2-00304867C16A.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/9CC4F1F6-8C8E-DF11-8F1C-00261894387D.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/9A09A901-8D8E-DF11-BEED-0018F3D09652.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/94012FC1-8C8E-DF11-8630-0018F3D095F8.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/92241FB0-8C8E-DF11-A95A-002618943852.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/906A9BC7-8C8E-DF11-9BCD-003048678B1C.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/829B15BC-8C8E-DF11-8070-0018F3D09650.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/7E2CE0B9-8C8E-DF11-A1E5-002618FDA265.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/76C170B5-8C8E-DF11-8F81-003048678FB4.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/6C8549AA-8C8E-DF11-A60E-00248C55CC9D.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/622365B4-8C8E-DF11-9064-0026189438D5.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/60BCB9AA-8C8E-DF11-BA7D-00261894386C.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/5A64E103-8D8E-DF11-90CE-001A92971B9C.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/587A8410-8D8E-DF11-BCE4-0018F3D0968E.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/565E7BBA-8C8E-DF11-B7AC-003048678FFE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/429709EE-8C8E-DF11-B547-0026189438F2.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/3E9D10B8-8C8E-DF11-A3DA-00304867BFA8.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/3C783627-8D8E-DF11-B318-00304867BFA8.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/38DF5419-8D8E-DF11-8F1C-0018F3D09652.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/388AFEB5-8C8E-DF11-8255-00261894394F.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/3857AEB6-8C8E-DF11-B6EE-001A92971BCA.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/34163655-8D8E-DF11-ABBD-002618FDA216.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/305A56B9-8C8E-DF11-BC96-002618FDA277.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/1E101FA2-8C8E-DF11-A857-003048679164.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/183F4E44-8D8E-DF11-884D-002618FDA277.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/120FC0BD-8C8E-DF11-A43C-002618943926.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/10E32DC4-8C8E-DF11-84DA-003048679214.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0004/0CF297B8-8C8E-DF11-99D8-0030486792DE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0001/06176787-A58B-DF11-966A-0030486790A6.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/DA03E474-6B8B-DF11-81EE-0018F3D096F8.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/D631829A-6E8B-DF11-9F2E-001A92810AD8.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/D0F7173E-6B8B-DF11-9C1F-0018F3D09704.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/BC8C20C8-6B8B-DF11-A3A5-001A92811744.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/94693D93-6F8B-DF11-9377-0018F3D09630.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/8E155992-6D8B-DF11-8BDF-0018F3D09676.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/8CB6A825-6E8B-DF11-9A84-001A92971BD6.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/86BDA75F-6B8B-DF11-9D17-002618943860.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/6AEADE03-6B8B-DF11-9265-001A92971B5E.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/641D12B7-6B8B-DF11-A454-00261894383F.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/62B05CC1-6C8B-DF11-BD85-0018F3D09636.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/4C731B08-6F8B-DF11-AEDF-0018F3D095EA.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/465B57A9-6E8B-DF11-A966-001A92810ADE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/0869056C-6A8B-DF11-BB73-00261894391F.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/042221AB-6E8B-DF11-A3ED-001A92810ACE.root',
        '/store/relval/CMSSW_3_8_0_pre8/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V6-v1/0000/02BD37AB-6E8B-DF11-9681-002354EF3BDB.root'
 
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


