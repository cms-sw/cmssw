import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.GlobalTag.globaltag = 'START36_V9::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
#       '/store/group/exotica/tadams/MinBias_TuneD6T_7TeV-pythia6/EXOHSCPSkimMinBiasD6TJun2010/941f63ecf717118dd5dd526fc7cfcc09/EXOHSCP_9_1_EJO.root',
#       '/store/group/exotica/tadams/MinBias_TuneD6T_7TeV-pythia6/EXOHSCPSkimMinBiasD6TJun2010/941f63ecf717118dd5dd526fc7cfcc09/EXOHSCP_99_1_SBC.root',
       '/store/group/exotica/MC_QCD80_V3/QCD_Pt80/HSCP_Skim_MC_LQ_QCD80V3/521d7f110f0e1d5e527904260b8c8fc5/EXOHSCP_66_1_p6q.root'
   )
)


########################################################################
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")

process.HSCPHLTFilter = cms.EDFilter("HSCPHLTFilter",
   TriggerProcess = cms.string("REDIGI36X"),
)


#process.TFileService = cms.Service("TFileService", 
#        fileName = cms.string('HSCP_tree.root')
#)

process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         "keep *_csc2DRecHits_*_*",
         "keep *_cscSegments_*_*",
         "keep *_dt1DRecHits_*_*",
         "keep *_rpcRecHits_*_*",
         "keep *_dt4DSegments_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep *_reducedHSCPhbhereco_*_*",
         "keep *_reducedHSCPEcalRecHitsEB_*_*",
         "keep *_TrackRefitter_*_*",
         "keep *_standAloneMuons_*_*",
         "keep *_globalMuons_*_*",
         "keep *_muonsSkim_*_*",
         "keep L1GlobalTriggerReadoutRecord_gtDigis_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep *_HSCParticleProducer_*_*",
    ),
    fileName = cms.untracked.string('HSCP.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('p1')
    ),
)

########################################################################

#LOOK AT SD PASSED PATH IN ORDER to avoid as much as possible duplicated events (make the merging of .root file faster)
process.p1 = cms.Path(process.HSCPHLTFilter * process.HSCParticleProducerSeq)
process.endPath = cms.EndPath(process.OUT)

