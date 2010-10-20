
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
photonValidation.OutputFileName = 'PhotonValidationRelVal390_QCD_Pt_80_120.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName


from Validation.RecoEgamma.tkConvValidator_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *
tkConversionValidation.OutputMEsInRootFile = True
tkConversionValidation.OutputFileName = 'ConversionValidationRelVal390_QCD_Pt_80_120.root'
conversionPostprocessing.batch = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName



process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0051/D67E68B4-3DD8-DF11-8E6E-0018F3D096BA.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0050/A09F61D7-FAD7-DF11-911D-00261894393C.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0050/A000259E-00D8-DF11-9078-0018F3D09648.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0050/94E6E313-01D8-DF11-A6F1-001A92971BD6.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0050/8C98D7F5-FED7-DF11-A87D-001A92971B96.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0050/7CB57EEA-FBD7-DF11-B192-0018F3D09676.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0050/6A9271E6-FBD7-DF11-B45F-0026189437EC.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-RECO/START39_V2-v1/0050/26A124E6-FBD7-DF11-B9B6-003048678DD6.root'


     ),
    
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/FECEBBE4-FBD7-DF11-9F97-002618943960.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/F81AA96B-FBD7-DF11-9A45-002618FDA259.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/F0FA7CE7-FBD7-DF11-8520-002618943921.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/EEE19FE4-FBD7-DF11-8D80-002618943880.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/C2E4C291-FFD7-DF11-B71B-0018F3D095F0.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/B8F4B669-FBD7-DF11-80E9-0018F3D096F0.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/B0720C16-01D8-DF11-87FA-002618943963.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/A25E6E22-00D8-DF11-83EB-00261894387A.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/A019B76B-FBD7-DF11-824A-003048678F84.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/9819C899-00D8-DF11-BB40-0018F3D096A0.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/84321DE5-FBD7-DF11-A077-002618FDA287.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/7A585EDC-FCD7-DF11-8BB4-002618943930.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/703842E2-FBD7-DF11-9163-002618943884.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/5C28A0F1-39D8-DF11-9396-0018F3D0961A.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/307992E6-FBD7-DF11-9BD2-001A92971B9A.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/2C37BED4-FAD7-DF11-BCBD-0026189437FA.root',
        '/store/relval/CMSSW_3_9_0/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START39_V2-v1/0050/2A5F791A-01D8-DF11-99EC-002618943900.root'
        
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


process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.tpSelecForEfficiency*process.tpSelecForFakeRate*process.tkConversionValidation*conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)


