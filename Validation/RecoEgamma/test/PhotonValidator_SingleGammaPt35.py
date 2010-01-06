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
process.GlobalTag.globaltag = 'MC_3XY_V14::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal350pre2_SingleGammaPt35.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 350pre2 single Photons pt=35GeV            
        '/store/relval/CMSSW_3_5_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V14-v1/0010/9640FF56-22EE-DE11-8CC3-002354EF3BDA.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V14-v1/0009/FCBE7F82-79ED-DE11-947F-00304867906C.root'

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
    
# official RelVal 350pre2 single Photons pt=35GeV

        '/store/relval/CMSSW_3_5_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0009/F8CFEF7D-79ED-DE11-9393-00261894391C.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0009/94E35537-22EE-DE11-97A8-00261894382D.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0009/86A39606-7AED-DE11-AF04-002618943947.root',
        '/store/relval/CMSSW_3_5_0_pre2/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0009/66E8E470-79ED-DE11-B840-003048678C62.root'
    

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
