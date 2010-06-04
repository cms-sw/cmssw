import FWCore.ParameterSet.Config as cms

process = cms.Process("EXOHSCPAnalysis")

# initialize MessageLogger and output report
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

#number of Events to be skimmed.
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'START3X_V26::All'
#process.GlobalTag.globaltag = 'GR_R_35X_V6::All'

#replace fileNames  with the file you want to skim
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_10_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_11_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_12_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_13_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_15_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_16_1.root',
        '/store/user/quertenmont/ExoticaSkim/HSCPSkim/querten/MinimumBias/EXOHSCPSkim_7TeV_356_RecoQuality_V2/6925c18577f7c0eecef10af135e8c3b9/EXOHSCP_17_1.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEFC70B6-F53D-DF11-B57E-003048679150.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FED8673E-F53D-DF11-9E58-0026189437EB.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEBF7874-EF3D-DF11-910D-002354EF3BDF.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FEA8ECD8-F13D-DF11-8EBD-00304867BFAE.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE838E9F-F43D-DF11-BEBA-00261894393B.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE7D760E-F43D-DF11-878A-00304867BED8.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FE2D63AD-F43D-DF11-B2B8-00261894395C.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC95A7F1-F13D-DF11-8C91-003048678C9A.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC5F5CA1-F53D-DF11-AFEE-002618FDA211.root',
        '/store/mc/Spring10/MinBias/GEN-SIM-RECO/START3X_V26A_356ReReco-v1/0009/FC140D7E-F43D-DF11-B6C2-0026189437ED.root',
   )
)

#process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducer_cff")         #IF RUNNING ON DIGI-RECO
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")

process.TFileService = cms.Service("TFileService",
        fileName = cms.string('tfile_out.root')
)

#Used for MC (Skimmed) or Data Sample
process.p = cms.Path(process.HSCParticleProducerSeq)# + process.HSCPTreeBuilder) #Uncomment the last part if you want to produce NTUPLE

#Used for Signal Sample
#process.p = cms.Path(process.genParticles + process.exoticaHSCPSeq + process.HSCParticleProducerSeq)# + process.HSCPTreeBuilder) #Uncomment the last part if you want to produce NTUPLE


process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep *_genParticles_*_*",
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
    fileName = cms.untracked.string('HSCP.root')
)

process.endPath = cms.EndPath(process.OUT)




