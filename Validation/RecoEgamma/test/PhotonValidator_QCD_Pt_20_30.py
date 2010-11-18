
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
process.GlobalTag.globaltag = 'START38_V7::All'


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

photonValidation.OutputFileName = 'PhotonValidationRelVal380_QCD_Pt_20_30.root'

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
# official RelVal 380 QCD_Pt_20_30
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0007/281BE2D2-4E96-DF11-A03D-00261894387D.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/C8157546-B395-DF11-A727-0030486790B0.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/ACCD0E43-B395-DF11-A2ED-002618FDA28E.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/A4BAE2E0-B195-DF11-97B9-002618FDA28E.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/9C588920-B395-DF11-94B9-00261894383B.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/8EF53BE5-B195-DF11-B8DE-002618943935.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/7AEB4A25-B395-DF11-88A2-003048678F84.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/4C3C4B43-B395-DF11-9657-003048678F92.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/24790B1E-B395-DF11-A73C-002618FDA204.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/240DDE48-B395-DF11-B170-003048679012.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0006/24036D41-B395-DF11-9100-003048678D78.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0005/88BEBB3E-B195-DF11-BF5C-0018F3D095EA.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0005/7E27A6CC-B195-DF11-AF06-002618943886.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-RECO/START38_V7-v1/0005/2031DBE6-B195-DF11-8603-00261894390E.root'
 
 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380 QCD_Pt_20_30
  '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0007/3C0EFCD4-4E96-DF11-8AB5-001A92971BBE.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/DCB36D3C-B395-DF11-BF6F-002618943856.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/DAFF9B36-B495-DF11-B1BC-001A92810AD4.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/CC8C6544-B395-DF11-92F8-003048678F84.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/C2B1E51C-B395-DF11-9DAB-003048679000.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/BEE226DD-B195-DF11-9F13-003048D15E02.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/B897CA24-B395-DF11-A681-003048678CA2.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/B2FAC6DB-B195-DF11-8F15-0026189438BD.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/B039E439-B395-DF11-A535-00261894390B.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/ACC53222-B395-DF11-B3DC-003048678AF4.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/A2796D1A-B395-DF11-B7EB-0026189438A5.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/9ECDFF3C-B395-DF11-AE70-003048678FF8.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/9E779944-B395-DF11-B732-003048678A80.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/9CE0BE36-B295-DF11-A837-00304867BECC.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/90D3121E-B395-DF11-8F95-00304867C0FC.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/86F5CDC9-B195-DF11-A5EC-00261894398D.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/867C1243-B395-DF11-A034-003048678F26.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/6EE13A23-B395-DF11-AA11-001A92810AE0.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/68393946-B395-DF11-8075-003048678F84.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/441C8B21-B395-DF11-822A-0026189438D8.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/36C53847-B395-DF11-B1C5-003048678FA6.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/2C2A253C-B395-DF11-B63C-002618943886.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/28232349-B395-DF11-BA4C-003048678FF4.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0006/2610123D-B395-DF11-8851-00261894394A.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0005/F46ED3D2-B195-DF11-8324-00304867926C.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0005/EA278D3D-B195-DF11-BA3B-001A92971AD0.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0005/A643ABDD-B195-DF11-99C1-00261894397D.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0005/A46A983F-B195-DF11-BF57-0018F3D096E6.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0005/5EB718E0-B195-DF11-8A19-002618943865.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0005/54C2873E-B195-DF11-BB81-0018F3D09702.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0005/4A2E2CD8-B195-DF11-84F4-00304867BECC.root',
        '/store/relval/CMSSW_3_8_0/RelValQCD_Pt_20_30/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V7-v1/0005/26741D3E-B195-DF11-B5C1-0018F3D0967A.root'
         
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


