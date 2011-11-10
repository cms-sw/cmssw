import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.GlobalTag.globaltag = 'START311_V2A::All'

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

process.GlobalTag.toGet = cms.VPSet(
   cms.PSet( record = cms.string('SiStripDeDxMip_3D_Rcd'),
            tag = cms.string('MC7TeV_Deco_3D_Rcd_38X'),
            connect = cms.untracked.string("sqlite_file:MC7TeV_Deco_SiStripDeDxMip_3D_Rcd.db")),
)

process.generalTracksSkim.filter       = cms.bool(True)
process.HSCParticleProducer.filter     = cms.bool(True)
process.HSCPTreeBuilder.reccordGenInfo = cms.untracked.bool(False)

process.dedxHarm2.calibrationPath      = cms.string("file:Gains.root")
process.dedxTru40.calibrationPath      = cms.string("file:Gains.root")
process.dedxProd.calibrationPath       = cms.string("file:Gains.root")
process.dedxASmi.calibrationPath       = cms.string("file:Gains.root")
process.dedxNPHarm2.calibrationPath    = cms.string("file:Gains.root")
process.dedxNPTru40.calibrationPath    = cms.string("file:Gains.root")
process.dedxNSHarm2.calibrationPath    = cms.string("file:Gains.root")
process.dedxNSTru40.calibrationPath    = cms.string("file:Gains.root")
process.dedxNPProd.calibrationPath     = cms.string("file:Gains.root")
process.dedxNPASmi.calibrationPath     = cms.string("file:Gains.root")

process.dedxHarm2.UseCalibration       = cms.bool(True)
process.dedxTru40.UseCalibration       = cms.bool(True)
process.dedxProd.UseCalibration        = cms.bool(True)
process.dedxASmi.UseCalibration        = cms.bool(True)
process.dedxNPHarm2.UseCalibration     = cms.bool(True)
process.dedxNPTru40.UseCalibration     = cms.bool(True)
process.dedxNSHarm2.UseCalibration     = cms.bool(True)
process.dedxNSTru40.UseCalibration     = cms.bool(True)
process.dedxNPProd.UseCalibration      = cms.bool(True)
process.dedxNPASmi.UseCalibration      = cms.bool(True)



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
#         "keep *_csc2DRecHits_*_*",
         "keep *_cscSegments_*_*",
#         "keep *_dt1DRecHits_*_*",
         "keep *_rpcRecHits_*_*",
         "keep *_dt4DSegments_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep *_reducedHSCPhbhereco_*_*",
         "keep *_reducedHSCPEcalRecHitsEB_*_*",
         "keep *_reducedHSCPEcalRecHitsEE_*_*",
         "keep *_TrackRefitter_*_*",
         "drop TrajectorysToOnerecoTracksAssociation_TrackRefitter__",
         "keep *_standAloneMuons_*_*",
         "drop recoTracks_standAloneMuons__*",
         "keep *_globalMuons_*_*",
         "keep *_muonsSkim_*_*",
#         "keep L1GlobalTriggerReadoutRecord_gtDigis_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep recoPFJets_ak5PFJets__*",
         "keep recoPFMETs_pfMet__*",
         "keep *_HSCParticleProducer_*_*",
         "keep *_HSCPIsolation01__*",
         "keep *_HSCPIsolation03__*",
         "keep *_HSCPIsolation05__*",
         "keep *_dedx*_*_HSCPAnalysis",
         "keep *_muontiming_*_HSCPAnalysis",
         "keep triggerTriggerEvent_hltTriggerSummaryAOD_*_*",
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

