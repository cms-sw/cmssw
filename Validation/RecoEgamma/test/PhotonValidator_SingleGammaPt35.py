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
process.GlobalTag.globaltag = 'MC_38Y_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre7_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V2-v1/0046/E0B026C1-B8D3-DF11-AAA8-001A92971AD0.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V2-v1/0043/FA539BB7-4FD3-DF11-8C7E-002618943875.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V2-v1/0043/6474F142-52D3-DF11-9450-0018F3D096CA.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0045/C634E695-B5D3-DF11-BF0A-0018F3D0967E.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0043/CE475D39-4ED3-DF11-A413-001A92810AD8.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0043/5AFA213E-50D3-DF11-A9C3-002618943834.root',
        '/store/relval/CMSSW_3_9_0_pre7/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V2-v1/0043/46019538-4FD3-DF11-B254-0026189437EB.root'


        
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
