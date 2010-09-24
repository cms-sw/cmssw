
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
process.GlobalTag.globaltag = 'START38_V12::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre4_QCD_Pt_80_120.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre4 QCD_Pt_80_120
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V11-v1/0027/EC2CDE75-75C3-DF11-8499-0018F3D096A2.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V11-v1/0025/C4D617F2-15C3-DF11-A39B-003048678A6C.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V11-v1/0025/8A9506F8-19C3-DF11-868E-001A92810AB8.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V11-v1/0025/86F4B760-13C3-DF11-8C5A-0018F3D09696.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V11-v1/0025/78F576E3-1AC3-DF11-A4D2-0018F3D096AE.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V11-v1/0025/62819AFA-14C3-DF11-89D0-0018F3D09620.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V11-v1/0025/3E4C3DE9-18C3-DF11-B8CF-00261894386F.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V11-v1/0025/1EE5BC72-14C3-DF11-9363-00261894388F.root'

     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre4 QCD_Pt_80_120
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0027/7CF1B567-75C3-DF11-A8A5-0018F3D096EE.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/FC70D9DE-12C3-DF11-9176-00304867BFB0.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/D2603CEB-14C3-DF11-93BD-003048D15CC0.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/CC63FC6F-14C3-DF11-9954-002618FDA237.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/C881E96D-14C3-DF11-9572-002618943975.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/BACFE65C-15C3-DF11-8311-003048678FB8.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/BACD81EC-14C3-DF11-99EF-00304866C398.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/9440B262-17C3-DF11-AF15-0018F3D096DC.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/6A5DC261-1AC3-DF11-ACFF-0018F3D0963C.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/5243CDE2-1AC3-DF11-A6C7-00304867903E.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/505D3AF4-15C3-DF11-83BB-0018F3C3E3A6.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/4EDD02FA-19C3-DF11-9C3D-0018F3D0963C.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/40D3F7EB-18C3-DF11-856A-001A9281172A.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/2400A162-1AC3-DF11-9863-001A92971BB8.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/20B30E65-19C3-DF11-BC4B-0018F3D096A2.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/10DEC0E3-13C3-DF11-B4F5-003048679294.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V11-v1/0025/0054B863-13C3-DF11-A930-00248C0BE013.root'
 
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


