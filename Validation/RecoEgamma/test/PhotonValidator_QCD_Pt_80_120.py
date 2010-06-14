
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
process.GlobalTag.globaltag = 'START37_V5::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre1_QCD_Pt_80_120.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre1 QCD_Pt_80_120
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V5-v1/0001/BCFFE72F-286E-DF11-8FF5-002618943838.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V5-v1/0001/08B06B96-FB6D-DF11-98A2-003048678BB8.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V5-v1/0000/E867991B-ED6D-DF11-8546-0018F3D09702.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V5-v1/0000/AE128B31-F26D-DF11-84E0-001A9281172C.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V5-v1/0000/80AEB898-EF6D-DF11-80CA-001A928116E0.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V5-v1/0000/7889A506-F46D-DF11-9739-003048679214.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V5-v1/0000/2E11EF1B-EE6D-DF11-B6D2-002618943868.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START37_V5-v1/0000/0A39BE2C-F16D-DF11-BFC0-001BFCDBD182.root'
 
     ),
    
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre1 QCD_Pt_80_120
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0001/F634EEB1-F46D-DF11-A4C6-001A92811728.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0001/EA7F3A25-286E-DF11-9D36-0018F3D096D8.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0001/2E33A7FB-FC6D-DF11-80D6-001A92971B80.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0001/02146399-F86D-DF11-8F63-0018F3D096E0.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/CEA96EB9-F26D-DF11-B7FD-001A92971B08.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/C8EDC506-ED6D-DF11-AE3C-003048678AC8.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/C499F2B9-F16D-DF11-A8B7-001A9281170C.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/BC551821-ED6D-DF11-A8EC-002618943866.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/80281131-EE6D-DF11-8295-001A9281171C.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/6E763DD2-F36D-DF11-AF96-001A928116FC.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/523A75B8-F16D-DF11-BB0C-003048678BE6.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/5093AD3D-F16D-DF11-B998-001A92810A9A.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/463763C4-ED6D-DF11-9130-001A92971BBE.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/44D58819-F06D-DF11-91EF-0030486790B0.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/2A5C921E-ED6D-DF11-8B4A-002354EF3BDC.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/1E5E71A7-EE6D-DF11-9A6B-003048679180.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START37_V5-v1/0000/04363022-F26D-DF11-949A-002618943882.root'
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


