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

process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducer_cff")           #IF RUNNING ON RAW-DIGI-RECO
#process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM


process.TFileService = cms.Service("TFileService",
        fileName = cms.string('tfile_out.root')
)

process.p = cms.Path(process.HSCParticleProducerSeq)

process.OUT = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring("drop *", "keep *_*_*_EXOHSCPAnalysis"),
    fileName = cms.untracked.string('out.root')
)

process.endPath = cms.EndPath(process.OUT)




