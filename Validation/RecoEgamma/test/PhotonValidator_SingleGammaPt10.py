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
process.GlobalTag.globaltag = 'MC_36Y_V3::All'


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
photonValidation.OutputFileName = 'PhotonValidationRelVal360pre4_SingleGammaPt10.root'

photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 360pre4 single Photons pt=10GeV
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V3-v1/0002/66BA26B0-1738-DF11-AAFF-0030487C5CE2.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt10/GEN-SIM-RECO/MC_36Y_V3-v1/0001/C84706C0-B137-DF11-A16A-0030487D0D3A.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 360pre4 single Photons pt=10GeV    
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0002/94C4422F-1738-DF11-8312-0030487A3C92.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0001/F23A7118-B337-DF11-86DB-00304879EDEA.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0001/503CD97C-B437-DF11-A926-0030487CD6D2.root',
        '/store/relval/CMSSW_3_6_0_pre4/RelValSingleGammaPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_36Y_V3-v1/0001/142A75F6-B037-DF11-8E52-00304879EE3E.root'
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



