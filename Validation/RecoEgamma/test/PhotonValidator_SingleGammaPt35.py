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
process.GlobalTag.globaltag = 'MC_38Y_V12::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre4_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre4 single Photons pt=35GeV
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V11-v1/0027/60693D70-75C3-DF11-8F97-0018F3D09648.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V11-v1/0025/F29C09FE-28C3-DF11-9825-001A92810AD8.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V11-v1/0025/8AF7B6FA-27C3-DF11-A3E7-001A928116AE.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre4 single Photons pt=35GeV
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V11-v1/0027/96296E71-75C3-DF11-B62F-0018F3D09660.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V11-v1/0025/ACC73679-28C3-DF11-B7AE-0018F3D09670.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V11-v1/0025/A2BE817E-27C3-DF11-B475-001A92811724.root',
        '/store/relval/CMSSW_3_9_0_pre4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V11-v1/0025/8C149EF9-27C3-DF11-BBA6-001A92971ACE.root'
        
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
