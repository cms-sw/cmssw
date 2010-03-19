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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre3_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(

# official RelVal 360pre3 single Photons pt=10GeV

        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V2-v1/0005/9C05E0C3-722F-DF11-870C-0018F3D095FE.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V2-v1/0005/404B2FFC-B02F-DF11-B33B-002618FDA21D.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre3 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V2-v1/0005/7A0D78E6-722F-DF11-8BE3-0018F3D09700.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V2-v1/0005/2099CDDC-722F-DF11-B356-001BFCDBD11E.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V2-v1/0005/1848F9EE-B02F-DF11-BB3B-002618943926.root',
        '/store/relval/CMSSW_3_6_0_pre3/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V2-v1/0005/108DF6D9-722F-DF11-97E9-0018F3D09678.root'


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



