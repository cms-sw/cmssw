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
photonValidation.OutputFileName = 'PhotonValidationRelVal350pre3_SingleGammaPt10.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 350pre3 single Photons pt=10GeV
        '/store/relval/CMSSW_3_5_0_pre3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_3XY_V15-v1/0006/443BED5E-D203-DF11-AD01-0030487CAF0E.root',
        '/store/relval/CMSSW_3_5_0_pre3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_3XY_V15-v1/0005/329B1F60-6503-DF11-A6DB-00304879FBB2.root'
        
    ),
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 350pre3 single Photons pt=10GeV    
 '/store/relval/CMSSW_3_5_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V15-v1/0005/F29388E6-6203-DF11-95A6-003048D2BC42.root',
 '/store/relval/CMSSW_3_5_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V15-v1/0005/36F75D3C-D103-DF11-9C22-0030487C7392.root',
 '/store/relval/CMSSW_3_5_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V15-v1/0005/2E356CF5-6503-DF11-9D16-0030487CD700.root',
 '/store/relval/CMSSW_3_5_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V15-v1/0005/26550554-6503-DF11-A6B0-0030487C60AE.root'
 
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



