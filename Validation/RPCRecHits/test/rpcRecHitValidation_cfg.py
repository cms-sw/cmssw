import FWCore.ParameterSet.Config as cms

process = cms.Process("RPCRecHitValidation")

### standard includes
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Services_cff")
process.load("Configuration.StandardSequences.Geometry_cff")

process.MessageLogger.cerr.FwkReport.reportEvery = 100

### conditions
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

### Load MC samples
import sys,os
if 'SAMPLE' in os.environ:
  sampleName = os.environ['SAMPLE']
  sys.path.append('./samples')
  sampleCfg = __import__(sampleName+"_cfg")
  process.source = sampleCfg.source
  process.maxEvents = sampleCfg.maxEvents
  process.GlobalTag.globaltag = sampleCfg.globaltag
else:
  sampleName = 'Test'
  process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0013/9263608A-6213-DF11-8763-001A92971AD0.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0012/CA67B049-3013-DF11-93F5-001731AF67E9.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0012/8C5E264D-3013-DF11-B544-002618943978.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0012/7E5C3C30-2E13-DF11-B3CC-001731AF669F.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-RECO/MC_3XY_V21-v1/0012/46AF6EC3-2F13-DF11-98A8-001A92971B88.root'
    ),
    secondaryFileNames = cms.untracked.vstring(
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0013/4A0D5688-6213-DF11-BB7C-0018F3D09600.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/F2EB355E-3013-DF11-A39B-001731AF68BF.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/F04A2A3F-3013-DF11-95E2-002354EF3BDD.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/B8FA88AC-2D13-DF11-97C0-0018F3D09630.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/AEABADC0-2F13-DF11-BD31-001A92971B88.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/AE412DB4-3013-DF11-A0F2-00304867C04E.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/AE35CB2E-2E13-DF11-8F83-001731AF669F.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/9A91EA42-3013-DF11-B7B0-0017319C92DA.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/9694AFC0-2F13-DF11-A76B-001A92971B88.root',
       '/store/relval/CMSSW_3_5_0/RelValSingleMuPt10/GEN-SIM-DIGI-RAW-HLTDEBUG/MC_3XY_V21-v1/0012/46E7724A-3013-DF11-81F6-001731AF67E9.root'
    )
  )
  process.GlobalTag.globaltag = 'MC_3XY_V21::All'
  process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

#process.options = cms.untracked.PSet(
#    wantSummary = cms.untracked.bool(True)
#)

### validation-specific includes
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load("DQMServices.Components.EDMtoMEConverter_cff")

process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = "/%s/%s/Validation" % (process.GlobalTag.globaltag.value()[:-5], sampleName)
process.DQMStore.verbose = 100
process.DQMStore.collateHistograms = False
process.dqmSaver.convention = 'Offline'
process.dqmSaver.saveByRun = cms.untracked.int32(-1)
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

process.options = cms.untracked.PSet(
    fileMode = cms.untracked.string('FULLMERGE')
)
#process.endjob_step = cms.Path(process.endOfProcess)
#process.MEtoEDMConverter_step = cms.Sequence(process.MEtoEDMConverter)

### User analyzers
#### Sim-Reco validation
process.load("Validation.RPCRecHits.rpcRecHitValidation_cfi")
process.rpcRecHitV.standAloneMode = True

#### RPCPorintProducer-Reco validation
process.load("RecoLocalMuon.RPCRecHit.rpcPointProducer_cfi")
process.load("Validation.RPCRecHits.rpcPointValidation_cfi")
process.dtVsRPCRecHitV.standAloneMode = True
process.cscVsRPCRecHitV.standAloneMode = True
process.trackVsRPCRecHitV.standAloneMode = True

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

process.p = cms.Path(process.rpcRecHitValidation_step+
                     process.rpcPointProducerPlusValidation_step*
                     process.postValidation_step)
#process.outPath = cms.EndPath(process.out)

