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
process.GlobalTag.globaltag = 'MC_37Y_V4::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal370pre5_SingleGammaPt10.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 370pre5 single Photons pt=10GeV
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt10/GEN-SIM-RECO/MC_37Y_V4-v1/0022/A8581847-4C63-DF11-85C7-00304867918A.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt10/GEN-SIM-RECO/MC_37Y_V4-v1/0022/58F8AF63-4D63-DF11-B8AA-003048D15DB6.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 370pre5 single Photons pt=10GeV
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0023/8E9DD1A8-8863-DF11-B0CB-003048679180.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0022/D43C44C0-4B63-DF11-B23A-002618943923.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0022/CC6D49D7-4C63-DF11-8116-0026189438E1.root',
        '/store/relval/CMSSW_3_7_0_pre5/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_37Y_V4-v1/0022/9CA1745C-4D63-DF11-B333-002354EF3BDD.root'

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



