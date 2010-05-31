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
process.GlobalTag.globaltag = 'MC_37Y_V4::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370 single Photons pt=35GeV

        '/store/relval/CMSSW_3_7_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V4-v1/0025/D4904A91-8869-DF11-8A4A-00261894383F.root',
        '/store/relval/CMSSW_3_7_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V4-v1/0025/A89BC3B0-5769-DF11-B4BA-0026189438FA.root',
        '/store/relval/CMSSW_3_7_0/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V4-v1/0025/5EC94F21-5469-DF11-B612-002618943970.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370 single Photons pt=35GeV
        '/store/relval/CMSSW_3_7_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0025/F22D3285-8869-DF11-9638-002618FDA265.root',
        '/store/relval/CMSSW_3_7_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0025/5068777D-5369-DF11-A361-002618943870.root',
        '/store/relval/CMSSW_3_7_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0025/16B6E189-5569-DF11-9BA3-00261894380A.root',
        '/store/relval/CMSSW_3_7_0/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0025/149E1B84-5469-DF11-BC91-00261894398C.root'
        
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
