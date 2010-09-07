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
process.GlobalTag.globaltag = 'MC_38Y_V9::All'

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
photonValidation.OutputFileName = 'PhotonValidationRelVal390pre3_SingleGammaPt35.root'


photonPostprocessing.batch = cms.bool(True)
photonPostprocessing.InputFileName = photonValidation.OutputFileName

process.source = cms.Source("PoolSource",
noEventSort = cms.untracked.bool(True),
duplicateCheckMode = cms.untracked.string('noDuplicateCheck'),
                            
    fileNames = cms.untracked.vstring(
# official RelVal 390pre3 single Photons pt=35GeV
        '/store/relval/CMSSW_3_9_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0021/045F68B9-74B6-DF11-BF26-002354EF3BCE.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0020/EABB1D3E-1BB6-DF11-BBDB-0026189438AC.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValSingleGammaPt35/GEN-SIM-RECO/MC_38Y_V9-v1/0020/02204431-26B6-DF11-B75C-001A92810AC0.root'

    ),
                            
    secondaryFileNames = cms.untracked.vstring(
# official RelVal 390pre3 single Photons pt=35GeV
        '/store/relval/CMSSW_3_9_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0021/F67FF8B6-74B6-DF11-84C2-0026189438B0.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/DECC73A6-25B6-DF11-82BE-0030486790B8.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0020/8E14BBB7-1BB6-DF11-8A0D-00304867BFBC.root',
        '/store/relval/CMSSW_3_9_0_pre3/RelValSingleGammaPt35/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_38Y_V9-v1/0019/6A64B549-09B6-DF11-BF04-002618943862.root'
        
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
