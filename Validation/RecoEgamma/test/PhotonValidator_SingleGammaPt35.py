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
process.GlobalTag.globaltag = 'MC_3XY_V15::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal350pre3_SingleGammaPt35.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 350pre3 single Photons pt=35GeV            
        '/store/relval/CMSSW_3_5_0_pre3/RelValSingleElectronPt35/GEN-SIM-RECO/MC_3XY_V15-v1/0005/88FF1000-F702-DF11-B13F-003048D2BE08.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValSingleElectronPt35/GEN-SIM-RECO/MC_3XY_V15-v1/0005/7C2AC862-D103-DF11-B64D-0030487CAF0E.root'

    ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(
    
# official RelVal 350pre3 single Photons pt=35GeV
        '/store/relval/CMSSW_3_5_0_pre3/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V15-v1/0006/3401F929-D203-DF11-B317-0030487CD162.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V15-v1/0005/AE565DFA-F602-DF11-852F-003048D37580.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V15-v1/0005/88F7ECA9-F602-DF11-A8D5-003048D374F2.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValSingleElectronPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V15-v1/0005/184EEC0A-F602-DF11-B1F9-001D09F29619.root'

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
