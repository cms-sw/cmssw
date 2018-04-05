import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCRecHitValidation")

### standard includes
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Services_cff")
#process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load('Configuration.StandardSequences.GeometryDB_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.RawToDigi_cff")

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.autoCond import autoCond
process.GlobalTag.globaltag = autoCond['startup']

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(),
    secondaryFileNames = cms.untracked.vstring()
)

### validation-specific includes
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.dqmSaver.convention = 'Offline'
#process.DQMStore.verbose = 100
process.DQMStore.collateHistograms = False
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE')
)
#process.endjob_step = cms.Path(process.endOfProcess)
#process.MEtoEDMConverter_step = cms.Sequence(process.MEtoEDMConverter)

### User analyzers
#### RPC Offline DQM
process.load("DQMOffline.Configuration.DQMOfflineMC_cff")
process.dqmSaver.workflow = '/RPC/MC/Validation'

#### Sim-Reco validation
process.load("Validation.RPCRecHits.rpcRecHitValidation_cfi")

#### RPCPorintProducer-Reco validation
process.load("RecoLocalMuon.RPCRecHit.rpcPointProducer_cff")
process.load("Validation.RPCRecHits.rpcPointValidation_cfi")

process.rpcPointProducerPlusValidation_step = cms.Sequence(
    process.rpcPointProducer*
    process.rpcPointVsRecHitValidation_step+
    process.simVsRPCPointValidation_step
)

#### Post validation steps
process.load("Validation.RPCRecHits.postValidation_cfi")

### Path
process.postValidation_step = cms.Sequence(
    process.rpcRecHitPostValidation_step+
    process.rpcPointVsRecHitPostValidation_step*
    process.dqmSaver
)

#process.out = cms.OutputModule("PoolOutputModule",
#                               outputCommands = cms.untracked.vstring('drop *', "keep *_*_*_RPCRecHitValidation", 
#                                                                      "keep *_MEtoEDMConverter_*_*"),
#                               fileName = cms.untracked.string('output.SAMPLE.root')
#)

process.p = cms.Path(
    process.RawToDigi*
    process.rpcTier0Source+
    process.rpcRecHitValidation_step+
    process.rpcPointProducerPlusValidation_step*
    process.postValidation_step
)
#process.outPath = cms.EndPath(process.out)

