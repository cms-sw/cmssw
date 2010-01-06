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
process.GlobalTag.globaltag = 'MC_31X_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal341_SingleGammaPt35.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 341 single Photons pt=35GeV            

        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V14-v1/0004/0E5F21BA-B6ED-DE11-BC4A-003048D2C1C4.root',
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt35/GEN-SIM-RECO/MC_3XY_V14-v1/0003/0213D232-6DED-DE11-B34F-001D09F251D1.root'

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 341 single Photons pt=35GeV

        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0004/08D970AB-B6ED-DE11-A109-003048D373AE.root',
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0003/DE3CFF45-6DED-DE11-8020-001D09F2516D.root',
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0003/A04A615A-6DED-DE11-A536-001D09F2AF96.root',
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0003/96EA7A86-69ED-DE11-A73A-003048D3756A.root'   

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
