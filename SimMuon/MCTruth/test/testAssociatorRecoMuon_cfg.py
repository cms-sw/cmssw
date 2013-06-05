import FWCore.ParameterSet.Config as cms

process = cms.Process("myana")

process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(100))

from Configuration.EventContent.EventContent_cff import *
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-RECO/START36_V7-v1/0020/B22217FE-455D-DF11-A958-0026189438FC.root',
    ),
    #secondaryFileNames = cms.untracked.vstring(
    #    '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/94F66BA6-425D-DF11-9798-0018F3D096AE.root',
    #    '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/8622F5DD-455D-DF11-9BFF-001A92971B8E.root',
    #    '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/70404F50-3F5D-DF11-8041-002618943843.root',
    #    '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/50656520-455D-DF11-8754-001A92971BC8.root',
    #    '/store/relval/CMSSW_3_6_1/RelValQCD_Pt_80_120/GEN-SIM-DIGI-RAW-HLTDEBUG/START36_V7-v1/0020/24F0268F-515D-DF11-AAF5-001A92810AA8.root',
    #)
)

# MessageLogger
process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.categories = cms.untracked.vstring('testAssociatorRecoMuon', 'MuonAssociatorByHits')
process.MessageLogger.cout = cms.untracked.PSet(
    noTimeStamps = cms.untracked.bool(True),
    threshold = cms.untracked.string('INFO'),
    INFO = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    default = cms.untracked.PSet(limit = cms.untracked.int32(0)),
    testAssociatorRecoMuon = cms.untracked.PSet(limit = cms.untracked.int32(10000000))
)
process.MessageLogger.cerr = cms.untracked.PSet(placeholder = cms.untracked.bool(True))

# Mixing Module
process.load("SimGeneral.MixingModule.mixNoPU_cfi")

# Standard Sequences
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.load("SimGeneral.TrackingAnalysis.trackingParticles_cfi")            # On RAW+RECO
#process.load("SimMuon.MCTruth.MuonAssociatorByHitsESProducer_cfi")           # On RAW+RECO
process.load("SimGeneral.TrackingAnalysis.trackingParticlesNoSimHits_cfi")    # On RECO
process.load("SimMuon.MCTruth.MuonAssociatorByHitsESProducer_NoSimHits_cfi")  # On RECO

process.GlobalTag.globaltag = cms.string('START37_V3::All')
process.options = cms.untracked.PSet(wantSummary = cms.untracked.bool(True))

# --- example Analyzer running MuonAssociatorByHits 
process.testanalyzer = cms.EDAnalyzer("testAssociatorRecoMuon",
    muonsTag  = cms.InputTag("muons"),
    trackType = cms.string("segments"),  # or 'inner','outer','global'
    #tpTag    = cms.InputTag("mix"),                          # RAW+RECO
    #associatorLabel = cms.string("muonAssociatorByHits"),            # RAW+RECO
    tpTag    = cms.InputTag("mergedtruthNoSimHits"),                # RECO Only
    associatorLabel = cms.string("muonAssociatorByHits_NoSimHits"), # RECO Only
) 

process.skim = cms.EDFilter("CandViewCountFilter", src = cms.InputTag("muons"), minNumber = cms.uint32(1))
process.test = cms.Path(process.skim+process.mix * process.trackingParticlesNoSimHits * process.testanalyzer) # RECO
#process.test = cms.Path(process.skim+process.mix * process.trackingParticles       * process.testanalyzer) # RAW+RECO

