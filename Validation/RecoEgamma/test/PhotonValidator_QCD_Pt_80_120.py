
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
process.GlobalTag.globaltag = 'START38_V9::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre2_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre2 QCD_Pt_80_120
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0015/24F54413-73A8-DF11-AD7F-002618943915.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0014/D2A1F82F-07A8-DF11-9873-001A92971B12.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0014/CAAE5D1A-12A8-DF11-859A-001A92971BD8.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0014/BAFA19A4-08A8-DF11-B96A-003048678BE6.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0014/8A0E3B9C-01A8-DF11-94AF-003048678FF8.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0014/7E4FC2FB-F9A7-DF11-9133-003048678A6A.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0014/6A4BF787-FBA7-DF11-B5BF-003048678E6E.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-RECO/START38_V9-v1/0014/02620F7E-F8A7-DF11-AC09-003048678B1C.root'

 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre2 QCD_Pt_80_120
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0015/101A8F08-6EA8-DF11-896B-0026189438F4.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/F2D7C69C-07A8-DF11-906E-003048678E94.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/F0B7C67D-F8A7-DF11-9779-00304867BFAE.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/F0ACC119-08A8-DF11-980A-0030486792BA.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/EA231FF2-F7A7-DF11-A3D5-003048678F02.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/E2240BF8-F8A7-DF11-9C06-00304867904E.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/C062BC7B-F8A7-DF11-91A9-001A92810ACA.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/8A69A384-FBA7-DF11-8444-003048678C9A.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/82A77283-FCA7-DF11-A0AA-003048678AFA.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/68A6B22B-07A8-DF11-8738-003048678F26.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/60ABC416-0DA8-DF11-AC5C-003048678B34.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/48F3FD17-11A8-DF11-B1BE-001A92971B28.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/349EDAA3-07A8-DF11-B386-003048678E80.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/243AB3E4-FDA7-DF11-92A3-003048678FB4.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/1EB0F0A0-02A8-DF11-BEE7-003048679296.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/14B20D15-15A8-DF11-A11E-003048678F62.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START38_V9-v1/0014/124A9DFA-FAA7-DF11-B12A-003048678F06.root'
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


