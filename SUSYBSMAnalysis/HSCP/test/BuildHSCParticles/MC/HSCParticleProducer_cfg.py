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
       '/store/group/exotica/QCD_Pt30/EXOHSCPSkim7TeV_Summer10_QCD_Pt30_START36_V9_S09-v1_GEN-SIM-RECODEBUG/ab8f9ae88c7562f56547014702e04e4c/EXOHSCP_9_1_IxU.root',
       '/store/group/exotica/QCD_Pt30/EXOHSCPSkim7TeV_Summer10_QCD_Pt30_START36_V9_S09-v1_GEN-SIM-RECODEBUG/ab8f9ae88c7562f56547014702e04e4c/EXOHSCP_8_1_rp5.root',
       '/store/group/exotica/QCD_Pt30/EXOHSCPSkim7TeV_Summer10_QCD_Pt30_START36_V9_S09-v1_GEN-SIM-RECODEBUG/ab8f9ae88c7562f56547014702e04e4c/EXOHSCP_6_1_8r0.root',
   )
)


########################################################################
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")

process.dedxCNPHarm2.calibrationPath   = cms.string("file:Gains.root")
process.dedxCNPTru40.calibrationPath   = cms.string("file:Gains.root")
process.dedxCNPMed.calibrationPath     = cms.string("file:Gains.root")
process.dedxProd.calibrationPath       = cms.string("file:Gains.root")
process.dedxSmi.calibrationPath        = cms.string("file:Gains.root")
process.dedxASmi.calibrationPath       = cms.string("file:Gains.root")
process.dedxSTCNPHarm2.calibrationPath = cms.string("file:Gains.root")
process.dedxSTCNPTru40.calibrationPath = cms.string("file:Gains.root")
process.dedxSTCNPMed.calibrationPath   = cms.string("file:Gains.root")
process.dedxSTProd.calibrationPath     = cms.string("file:Gains.root")
process.dedxSTSmi.calibrationPath      = cms.string("file:Gains.root")
process.dedxSTASmi.calibrationPath     = cms.string("file:Gains.root")

process.dedxCNPHarm2.UseCalibration    = cms.bool(True) 
process.dedxCNPTru40.UseCalibration    = cms.bool(True)
process.dedxCNPMed.UseCalibration      = cms.bool(True)
process.dedxProd.UseCalibration        = cms.bool(True)
process.dedxSmi.UseCalibration         = cms.bool(True)
process.dedxASmi.UseCalibration        = cms.bool(True)
process.dedxSTCNPHarm2.UseCalibration  = cms.bool(True)
process.dedxSTCNPTru40.UseCalibration  = cms.bool(True)
process.dedxSTCNPMed.UseCalibration    = cms.bool(True)
process.dedxSTProd.UseCalibration      = cms.bool(True)
process.dedxSTSmi.UseCalibration       = cms.bool(True)
process.dedxSTASmi.UseCalibration      = cms.bool(True)


process.HSCPHLTFilter = cms.EDFilter("HSCPHLTFilter",
   TriggerProcess = cms.string("HLT"),
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

