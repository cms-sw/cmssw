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
photonValidation.OutputFileName = 'PhotonValidationRelVal341_SingleGammaPt10.root'
photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

    
# official RelVal 341 single Photons pt=10GeV
        
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_3XY_V14-v1/0004/AA93E67F-7EED-DE11-A0E9-000423D98BC4.root',
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt10/GEN-SIM-RECO/MC_3XY_V14-v1/0004/46D8A4B3-B5ED-DE11-B498-0030487C6062.root'
   
 
    ),
                            
    secondaryFileNames = cms.untracked.vstring(

# official RelVal 341 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0004/D8C89C7E-7FED-DE11-8F2D-0030486730C6.root',
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0004/36905F1F-7EED-DE11-AAA5-001D09F2A49C.root',
        '/store/relval/CMSSW_3_4_1/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V14-v1/0004/0C7EBB39-80ED-DE11-A922-0030486730C6.root'
    
    
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



