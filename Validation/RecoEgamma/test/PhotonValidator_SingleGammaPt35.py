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
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'MC_39Y_V6::All'

process.DQMStore = cms.Service("DQMStore");
process.load("DQMServices.Components.DQMStoreStats_cfi")
from DQMServices.Components.DQMStoreStats_cfi import *
dqmStoreStats.runOnEndJob = cms.untracked.bool(True)



process.maxEvents = cms.untracked.PSet(
#input = cms.untracked.int32(10)
)



from Validation.RecoEgamma.photonValidationSequence_cff import *
from Validation.RecoEgamma.photonPostprocessing_cfi import *
from Validation.RecoEgamma.conversionPostprocessing_cfi import *


photonValidation.OutputFileName = 'PhotonValidationRelVal394_SingleGammaPt35.root'

photonPostprocessing.standalone = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName
photonPostprocessing.OuputFileName = photonValidation.OutputFileName

conversionPostprocessing.standalone = cms.bool(True)
conversionPostprocessing.InputFileName = tkConversionValidation.OutputFileName
conversionPostprocessing.OuputFileName = tkConversionValidation.OutputFileName


process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V6-v1/0001/94B0B227-1BF8-DF11-AAA4-001A92971BDA.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V6-v1/0001/4624F327-19F8-DF11-B4A5-00261894380D.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt35/GEN-SIM-RECO/MC_39Y_V6-v1/0001/4228EBB7-36F8-DF11-A627-0026189437F2.root'
    ),
                            
    secondaryFileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V6-v1/0002/3CE375CF-39F8-DF11-B060-002354EF3BE0.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V6-v1/0001/F249998C-19F8-DF11-8295-0030486792B4.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V6-v1/0001/E6A1119B-19F8-DF11-87DC-0018F3D09702.root',
        '/store/relval/CMSSW_3_9_4/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_39Y_V6-v1/0000/BC4E76EE-E8F7-DF11-9C9E-002618943964.root'


        
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



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.conversionPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)
