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
process.GlobalTag.globaltag = 'MC_37Y_V5::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal380pre1_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 380pre1 single Photons pt=35GeV

        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V5-v1/0001/566EC8F1-276E-DF11-9F3E-001A928116C0.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V5-v1/0000/4E7827BC-DE6D-DF11-ADC9-003048678F84.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_37Y_V5-v1/0000/064F190E-E96D-DF11-B360-00248C0BE014.root'
 



    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 380pre1 single Photons pt=35GeV
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V5-v1/0001/38A46D02-286E-DF11-BAFB-001A928116F8.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V5-v1/0000/BE4AB6B7-DF6D-DF11-86F1-002618943809.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V5-v1/0000/84CDCA1A-DE6D-DF11-BB38-002618943949.root',
        '/store/relval/CMSSW_3_8_0_pre1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V5-v1/0000/5E62A0A8-DE6D-DF11-9CC3-0030486791DC.root'
        
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
