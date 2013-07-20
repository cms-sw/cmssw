import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.load("Configuration.StandardSequences.Reconstruction_cff")

#process.load("RecoVertex.BeamSpotProducer.BeamSpotEarly10TeVCollision_cfi")
#process.load("RecoVertex.BeamSpotProducer.BeamSpotFakeConditionsEarly10TeVCollision_cff")
#process.load("RecoVertex.Configuration.RecoVertex_cff");


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 10


process.GlobalTag.globaltag = 'START36_V9::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/results/exotica/EXO_HSCP_gluino600_361p4_GEN_SIM/StoreResults-EXO_HSCPgluino600_361p4_GEN_onlyneutralSIM_RECO-bc6efd9c361cd5f72b4a0d59e9beaa8c/EXO_HSCP_gluino600_361p4_GEN_SIM/USER/StoreResults-EXO_HSCPgluino600_361p4_GEN_onlyneutralSIM_RECO-bc6efd9c361cd5f72b4a0d59e9beaa8c/0000/7220D22C-ADB7-DF11-B316-001EC9AA9A09.root',
        '/store/results/exotica/EXO_HSCP_gluino600_361p4_GEN_SIM/StoreResults-EXO_HSCPgluino600_361p4_GEN_onlyneutralSIM_RECO-bc6efd9c361cd5f72b4a0d59e9beaa8c/EXO_HSCP_gluino600_361p4_GEN_SIM/USER/StoreResults-EXO_HSCPgluino600_361p4_GEN_onlyneutralSIM_RECO-bc6efd9c361cd5f72b4a0d59e9beaa8c/0000/1841DA2B-ADB7-DF11-8731-001EC9AAA030.root'
#'/store/results/exotica/EXO_HSCP_gluino200_361p4_GEN_SIM/StoreResults-EXO_HSCPgluino200_361p4_GEN_onlyneutralSIM_RECO-bc6efd9c361cd5f72b4a0d59e9beaa8c/EXO_HSCP_gluino200_361p4_GEN_SIM/USER/StoreResults-EXO_HSCPgluino200_361p4_GEN_onlyneutralSIM_RECO-bc6efd9c361cd5f72b4a0d59e9beaa8c/0000/B2CE7BF0-4BB8-DF11-A174-001EC9AA99A5.root',
#'/store/results/exotica/EXO_HSCP_gluino200_361p4_GEN_SIM/StoreResults-EXO_HSCPgluino200_361p4_GEN_onlyneutralSIM_RECO-bc6efd9c361cd5f72b4a0d59e9beaa8c/EXO_HSCP_gluino200_361p4_GEN_SIM/USER/StoreResults-EXO_HSCPgluino200_361p4_GEN_onlyneutralSIM_RECO-bc6efd9c361cd5f72b4a0d59e9beaa8c/0000/3C0100D8-4BB8-DF11-9795-001EC9AAA003.root',

   )
)

########################################################################
process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)

process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi')
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")

process.generalTracksSkim.filter       = cms.bool(False) 
process.HSCParticleProducer.filter     = cms.bool(False)
process.HSCPTreeBuilder.reccordGenInfo = cms.untracked.bool(True)

################## DEDX ANALYSIS SEQUENCE MODULES ##################

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

########################################################################

process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_genParticles_*_*",
         "keep *_offlinePrimaryVertices_*_HSCPAnalysis",
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
         "keep triggerTriggerEvent_hltTriggerSummaryAOD_*_*",
    ),
    fileName = cms.untracked.string('HSCP.root')
)

########################################################################

process.rereco = cms.Path(process.siPixelRecHits + process.siStripMatchedRecHits + process.offlineBeamSpot + process.recopixelvertexing + process.ckftracks + process.vertexreco);
process.p = cms.Path(process.genParticles + process.exoticaHSCPSeq + process.HSCParticleProducerSeq)

process.outpath  = cms.EndPath(process.OUT)
process.schedule = cms.Schedule(process.rereco, process.p, process.outpath)
