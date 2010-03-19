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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre3_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre3 single Photons pt=35GeV            
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V2-v1/0005/7EE6FA0B-B12F-DF11-B967-00304867BFC6.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_36Y_V2-v1/0004/4E96FB77-5B2F-DF11-A1AF-001A92971AEC.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(

        # official RelVal 360pre3 single Photons pt=35GeV
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V2-v1/0005/F623F3FE-B02F-DF11-82B3-00304867924E.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V2-v1/0004/F855B176-5B2F-DF11-BD40-001A92811744.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V2-v1/0004/A87D7012-5B2F-DF11-8CD3-001A92811744.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V2-v1/0004/129B5A5D-5D2F-DF11-BF16-00304867920A.root'

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
