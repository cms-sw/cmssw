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
process.GlobalTag.globaltag = 'MC_37Y_V2::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre3_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre3 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V2-v1/0019/94FC5CBA-0058-DF11-8DD7-0018F3D09612.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V2-v1/0019/80755AA5-FE57-DF11-9E17-001A92971B56.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V2-v1/0019/0EE5A19D-3458-DF11-A03C-003048678A6A.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre3 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V2-v1/0019/F4FAFE28-0058-DF11-B924-002618FDA216.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V2-v1/0019/B8E9F11F-FF57-DF11-AB8B-001A92810AC6.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V2-v1/0019/AA008020-FE57-DF11-BC33-003048678F06.root',
        '/store/relval/CMSSW_3_7_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V2-v1/0019/008765FA-3458-DF11-85B8-003048678A6A.root'
    
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
