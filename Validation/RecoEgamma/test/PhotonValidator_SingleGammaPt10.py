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
photonValidation.OutputFileName = 'PhotonValidationRelVal335_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
        
# official RelVal 335 single Photons pt=10GeV

        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V9-v1/0008/F6BE095E-12DC-DE11-9566-0018F3D0969C.root',
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt10/GEN-SIM-RECO/MC_31X_V9-v1/0007/F00BA778-CDDB-DE11-805F-001A92810ADE.root'
        
        ),
                            
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 335 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/F4B33A75-CDDB-DE11-91A2-0018F3D095F0.root',
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/780327D3-D7DB-DE11-9070-003048678C9A.root',
        '/store/relval/CMSSW_3_3_5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_31X_V9-v1/0007/7284CC76-CDDB-DE11-8985-001A92971B3C.root'

    
    )
 )


photonPostprocessing.rBin = 48

## For single gamma pt =10
photonValidation.eMax  = 100
photonValidation.etMax = 50
photonValidation.etScale = 0.20
photonPostprocessing.eMax  = 100
photonPostprocessing.etMax = 50



process.FEVT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("keep *_MEtoEDMConverter_*_*"),
    fileName = cms.untracked.string('pippo.root')
)



process.p1 = cms.Path(process.tpSelection*process.photonValidationSequence*process.photonPostprocessing*process.dqmStoreStats)
process.schedule = cms.Schedule(process.p1)



