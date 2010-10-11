
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
process.GlobalTag.globaltag = 'START39_V2::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre7_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0048/666155C9-03D5-DF11-9E9E-0026189438D9.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/F8678D1E-A7D4-DF11-9887-002354EF3BE0.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/DE34A8D4-A5D4-DF11-8BAB-00261894390E.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/C634F8A1-A5D4-DF11-A407-002618943913.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/B89F8E20-A2D4-DF11-91A4-00304867C1B0.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/AE561FA3-A5D4-DF11-B414-002618943834.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/A6AEE1A2-A2D4-DF11-B850-00248C55CC62.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/98F49E47-A4D4-DF11-9884-003048678B7C.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/96E1599F-A6D4-DF11-8134-003048678FE0.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/76F1A1BA-A3D4-DF11-A935-003048678B3C.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/6EC4A5B9-A3D4-DF11-AAAD-003048D15DDA.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/5C7B519D-A6D4-DF11-88EE-003048679044.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/52A245B3-A7D4-DF11-9FB2-002354EF3BCE.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/52421ABA-A4D4-DF11-AD32-0030486792BA.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/501B25CB-A2D4-DF11-8A6F-003048678B12.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/028A8ACC-A2D4-DF11-B5BF-00261894394F.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0047/009326A4-A5D4-DF11-B02C-0026189438D7.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0045/A263B12F-B4D3-DF11-BE9F-00261894396A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0044/B0A1717B-66D3-DF11-AF07-002618943962.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0044/A6508E83-65D3-DF11-BE85-002618943971.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0044/9222E6F0-64D3-DF11-856F-003048679076.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0044/661C235B-63D3-DF11-898F-001A92811726.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0044/48637DE7-67D3-DF11-B5E1-00304867C1B0.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0044/2AFB6A60-64D3-DF11-AB04-003048678AE4.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0044/1C310575-6AD3-DF11-BF6E-0026189438A5.root'


     ),
    
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0048/8A53BDC3-03D5-DF11-968E-0026189438D9.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/FEBA4D9C-A6D4-DF11-8397-002618943985.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/F60841D2-A5D4-DF11-A586-003048678F78.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/F4431AA2-A5D4-DF11-8F1C-0026189438D7.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/F25AA1A1-A5D4-DF11-84C7-002618943944.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/E24AD54D-A4D4-DF11-B6D7-002618943976.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/D6861BB9-A4D4-DF11-9F30-002618943950.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/C4A5CAB3-A7D4-DF11-AEDB-003048679046.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/BC2C841C-A7D4-DF11-9FB0-00261894394D.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/B8ACB2A0-A5D4-DF11-A631-002618943868.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/B670809C-A6D4-DF11-95AB-00261894397A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/AE915DB8-A3D4-DF11-95E4-00261894387D.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/AE141DB7-A3D4-DF11-8B80-003048D15DDA.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/ACFBE795-A1D4-DF11-97E0-002618943826.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/AC67B7A0-A6D4-DF11-A086-00261894392F.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/A6C064A1-A5D4-DF11-B041-00261894380B.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/A46207A0-A5D4-DF11-9C52-00261894393A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/A03F3C96-A1D4-DF11-B962-002618943896.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/90B8CBD0-A5D4-DF11-B20C-003048678C26.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/8A40D8B7-A3D4-DF11-92BA-00261894384A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/8060CAB8-A4D4-DF11-B7E0-00304867BECC.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/803DD3CA-A2D4-DF11-A560-0026189437F5.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/788C95A0-A5D4-DF11-AF5D-003048678FC4.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/687EDC9E-A6D4-DF11-BF6A-002618FDA26D.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/6694CCA0-A5D4-DF11-B806-002618943982.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/5C31EBC9-A2D4-DF11-933A-003048678F0C.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/5AF23FCA-A2D4-DF11-84D3-002354EF3BDF.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/4ADC61C9-A2D4-DF11-BE63-002618943985.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/4075B148-A4D4-DF11-A854-002618943922.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/3882669B-A6D4-DF11-A64D-00304867918A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/2E43929F-A6D4-DF11-B530-003048678BB2.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/2C99D84D-A4D4-DF11-AB7A-002618943866.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/2435AA9A-A6D4-DF11-B3A9-002618943920.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/22BE42A2-A2D4-DF11-9DCA-00248C0BE01E.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/1E0834B1-A7D4-DF11-9B4A-003048678D52.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/1C7178C9-A2D4-DF11-8E88-00261894390C.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/18F0ACC9-A2D4-DF11-91B2-002618943854.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/1004C8CA-A2D4-DF11-AE31-002618943950.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/0CDFD51D-A7D4-DF11-88DF-003048678E8A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/0A216FA3-A2D4-DF11-A0A7-003048678B1A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/06982CA2-A5D4-DF11-8619-002618943922.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0047/001129B7-A3D4-DF11-8C37-00304867BFB2.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0045/C26993AD-B5D3-DF11-9CF2-00261894391F.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/FC3B98E7-65D3-DF11-9461-0030486792B8.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/CEB39B72-6AD3-DF11-BF52-0026189438AC.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/AA37DAE9-63D3-DF11-A6FB-003048D42D92.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/8C19F56C-66D3-DF11-9CFC-002618943845.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/8A805783-65D3-DF11-A2F1-003048678AE4.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/7ACB1508-65D3-DF11-9DC9-001A92811726.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/7A3D856E-68D3-DF11-AFF9-002354EF3BE4.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/5E77885C-63D3-DF11-B98E-0018F3D0969C.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/5AB29AF0-67D3-DF11-B82E-002618943919.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/4CB9825E-64D3-DF11-926F-002618943856.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/40CC955D-64D3-DF11-9ABE-002354EF3BDB.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/3A7C865E-64D3-DF11-ADC6-001A928116AE.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/38ACE059-63D3-DF11-946D-0018F3D09616.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/34D13B7A-67D3-DF11-A869-003048678D9A.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/321A10F1-66D3-DF11-AB71-002618943826.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0044/28E2B282-65D3-DF11-97CF-002618943865.root'



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


