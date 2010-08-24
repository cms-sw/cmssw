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
process.GlobalTag.globaltag = 'MC_38Y_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre2_SingleGammaPt35TEST.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre2 single Photons pt=35GeV
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0015/0ED4F106-6EA8-DF11-B9F8-002618943945.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0014/A2386D6E-F2A7-DF11-BC8D-001BFCDBD176.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0014/3EBBBB9B-02A8-DF11-BECE-0018F3D096EE.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre2 single Photons pt=35GeV
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0015/A2F5DF08-73A8-DF11-B35E-002618943870.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0014/D2D237E9-F1A7-DF11-B25E-00304867BFAA.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0014/A82C9175-F5A7-DF11-862E-003048D3C010.root',
        '/store/relval/CMSSW_3_9_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0014/12D39BF5-04A8-DF11-902D-0018F3D096BE.root'
        
    )
 )


photonPostprocessing.rBin = 48

## For single gamma pt = 35
photonValidation.eMax  = 300
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonValidation.dCotCutOn = False
photonValidation.dCotCutValue = 0.15

process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
