import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Era_Phase2_cff import Phase2
process = cms.Process('recHitTest',Phase2)

#process.MessageLogger = cms.Service("MessageLogger",
#    debugModules = cms.untracked.vstring('siPixelRawData'),
#    destinations = cms.untracked.vstring("cout"),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('ERROR')
#    )
#)
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
process.load('Configuration.StandardSequences.L1Reco_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.RecoSim_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10)
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

# list of files

process.source = cms.Source("PoolSource",
    fileNames =  cms.untracked.vstring(
        'file:/eos/cms/store/relval/CMSSW_11_2_0_pre5/RelValTTbar_14TeV/GEN-SIM-RECO/PU25ns_110X_mcRun4_realistic_v3_2026D49PU200-v1/20000/CA4DC8DF-AAEC-DA43-931B-BF234D979A9D.root'
      )
)
# Production Info
process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string('$Revision: 1.19 $'),
    annotation = cms.untracked.string('step1 nevts:1'),
    name = cms.untracked.string('Applications')
)
# Output definition

process.DQMoutput = cms.OutputModule("PoolOutputModule",
    splitLevel = cms.untracked.int32(0),
    outputCommands = process.DQMEventContent.outputCommands,
    fileName = cms.untracked.string('step1_rechitTest.root'),
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string(''),
        dataTier = cms.untracked.string('')
    )
)

process.load('RecoLocalTracker.SiPhase2Clusterizer.phase2TrackerClusterizer_cfi')
process.load('RecoLocalTracker.Phase2TrackerRecHits.Phase2StripCPEESProducer_cfi')
process.load('RecoLocalTracker.Phase2TrackerRecHits.Phase2TrackerRecHits_cfi')
process.load('RecoLocalTracker.SiPixelRecHits.SiPixelRecHits_cfi')
#process.rechits_step = cms.Path(process.siPhase2RecHits * process.siPixelRecHits)
process.rechits_step = cms.Path(process.siPixelRecHits)


#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(8)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)

#process.load('Validation.SiTrackerPhase2V.Phase2OTValidateRecHit_cfi')
process.load('Validation.SiTrackerPhase2V.Phase2ITValidateRecHit_cfi')

#process.rechitval_seq = cms.Sequence(process.rechitValidIT)

process.load('DQMServices.Components.DQMEventInfo_cfi')
#process.dqmEnv.subSystemFolder = cms.untracked.string('Ph2Tk')
process.dqm_comm = cms.Sequence(process.dqmEnv)

# Path and EndPath definitions
process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.DQMoutput)
'''
process.Timing = cms.Service("Timing",
  summaryOnly = cms.untracked.bool(False),
  useJobReport = cms.untracked.bool(True)
)

process.SimpleMemoryCheck = cms.Service("SimpleMemoryCheck",
    ignoreTotal = cms.untracked.int32(1),
    moduleMemorySummary = cms.untracked.bool(True)
)
'''
process.p = cms.Path(process.siPixelRecHits 
                     #*process.siPhase2RecHits 
                     * process.rechitValidIT 
                     #* process.rechitValidOT 
                     * process.dqm_comm )
