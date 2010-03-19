
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
process.GlobalTag.globaltag = 'MC_36Y_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre3_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre3 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V2-v1/0005/6C47CD03-B12F-DF11-8E56-003048678FB4.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V2-v1/0004/FAF77CD8-662F-DF11-BD1A-001A92971B9C.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V2-v1/0004/D6D29373-672F-DF11-88BA-0018F3D095F8.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V2-v1/0004/B02F76EA-682F-DF11-9091-001A92971BB2.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V2-v1/0004/A2EFE88E-682F-DF11-A2D0-001A92971B26.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V2-v1/0004/8261EA5A-672F-DF11-970C-00304866C398.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V2-v1/0004/6ED8A370-672F-DF11-BAEE-001A92810AEC.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V2-v1/0004/26FB3685-672F-DF11-AE1E-001A92971B0E.root'
        
        ),
    
    secondaryFileNames = cms.untracked.vstring(
        # official RelVal 360pre3 QCD_Pt_80_120
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0005/AE5D71F3-B02F-DF11-B2DF-003048678AE4.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/ECAF24D7-662F-DF11-8E2E-003048679006.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/EC0DC685-682F-DF11-BAEA-003048678FE6.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/DEDA8A16-682F-DF11-957E-001A928116EE.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/D4E03178-682F-DF11-8EFB-001A92971BC8.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/BE9D9D52-672F-DF11-9F79-001A92810ACA.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/BC626464-672F-DF11-9686-001A9281174C.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/982E0A67-672F-DF11-8820-0018F3D0960C.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/82245C6C-672F-DF11-B845-003048678FF8.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/62467B5F-672F-DF11-BE33-001A92971B26.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/5E6692D7-662F-DF11-BA0B-00304867C026.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/5C720419-682F-DF11-9AA4-0018F3D096B4.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/5889A952-672F-DF11-BC4E-00304867902C.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/569EEC56-672F-DF11-9F31-001BFCDBD166.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/2ED874D6-662F-DF11-A35B-001A92971B30.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/22D3BB6D-672F-DF11-A853-003048678A76.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V2-v1/0004/123F93E8-682F-DF11-BCA7-001A92971B0E.root'


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


