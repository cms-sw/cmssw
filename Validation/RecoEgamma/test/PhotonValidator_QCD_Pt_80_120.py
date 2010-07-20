
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
process.GlobalTag.globaltag = 'START38_V4::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre7_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre7 QCD_Pt_80_120

        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V4-v1/0002/C65C66C6-3486-DF11-B721-0030487CD77E.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V4-v1/0002/B8922660-8086-DF11-B5E1-003048F1BF68.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V4-v1/0002/A8F811D5-3386-DF11-AAD8-0030487CD76A.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V4-v1/0002/92DE21C7-2F86-DF11-A1ED-0030487CD7EA.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V4-v1/0002/901758CC-3086-DF11-A5D8-0030487CD178.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V4-v1/0002/82EA557A-3386-DF11-B781-0030487CD704.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V4-v1/0002/7EA838D5-3186-DF11-9959-0030487A3C9A.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V4-v1/0002/6CD3CDAC-3386-DF11-9F95-0030487CD716.root'

 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre7 QCD_Pt_80_120

        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/EE928ACF-3086-DF11-85B4-0030487C90EE.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/E2BAF529-3186-DF11-9120-0030487A3232.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/CC71E3AD-3286-DF11-A333-0030487C608C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/C8846F17-2E86-DF11-BA8C-0030487A3232.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/C292C9D4-3186-DF11-AC9C-0030487CD17C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/C010C1A0-3386-DF11-8C5C-0030487C5CFA.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/9A2C88C4-3486-DF11-953D-0030487D05B0.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/8C1C9FD8-3186-DF11-9C05-0030487A18F2.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/864381AB-3386-DF11-A264-0030487CD17C.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/7A2B8228-3486-DF11-ADB4-0030487CD7EE.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/6E5F155D-8086-DF11-9BCE-003048F11DE2.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/5C2B28A1-3386-DF11-8AD0-0030487A3232.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/5072A3C7-3386-DF11-8BE8-0030487C5CFA.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/48AAAFA4-3386-DF11-BACA-0030487CD7EE.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/467B8DB0-3486-DF11-BD2F-0030487CD7EE.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/2EEFB340-3186-DF11-9AF9-0030487CD6F2.root',
        '/store/relval/CMSSW_3_8_0_pre7/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V4-v1/0002/1AEB090A-2F86-DF11-A3CA-0030487C608C.root'
 
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


