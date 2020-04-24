import FWCore.ParameterSet.Config as cms
process = cms.Process("MergeHLT")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load("FWCore.MessageService.MessageLogger_cfi")
from SUSYBSMAnalysis.HSCP.HSCPVersion_cff import *

process.MessageLogger.cerr.FwkReport.reportEvery = 5000
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
XXX_INPUT_XXX
   )
)

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.HSCPHLTDuplicate = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(True),
   TriggerProcess   = cms.string("HLT"),
   MuonTrigger1Mask    = cms.int32(0),  #Activated
   PFMetTriggerMask    = cms.int32(0),  #Activated
   L2MuMETTriggerMask  = cms.int32(0),
)
process.DuplicateFilter = cms.Path(process.HSCPHLTDuplicate   )


process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.HSCPHLTTriggerMuDeDx = process.hltHighLevel.clone()
process.HSCPHLTTriggerMuDeDx.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
process.HSCPHLTTriggerMuDeDx.andOr = cms.bool( True ) #OR
process.HSCPHLTTriggerMuDeDx.throw = cms.bool( False )
process.HSCPHLTTriggerMuDeDx.HLTPaths = ["HLT_Mu*_dEdx*"]
process.HSCPHLTTriggerMuDeDxFilter = cms.Path(process.HSCPHLTTriggerMuDeDx   )

process.HSCPHLTTriggerMetDeDx = process.HSCPHLTTriggerMuDeDx.clone() 
process.HSCPHLTTriggerMetDeDx.HLTPaths = ["HLT_MET*_dEdx*"]
process.HSCPHLTTriggerMetDeDxFilter = cms.Path(process.HSCPHLTTriggerMetDeDx   )

process.HSCPHLTTriggerHtDeDx = process.HSCPHLTTriggerMuDeDx.clone()
process.HSCPHLTTriggerHtDeDx.HLTPaths = ["HLT_HT*_dEdx*"]
process.HSCPHLTTriggerHtDeDxFilter = cms.Path(process.HSCPHLTTriggerHtDeDx   )

process.HSCPHLTTriggerMu = process.HSCPHLTTriggerMuDeDx.clone()
process.HSCPHLTTriggerMu.HLTPaths = ["HLT_Mu40_*"]
process.HSCPHLTTriggerMuFilter = cms.Path(process.HSCPHLTTriggerMu   )

process.HSCPHLTTriggerMet = process.HSCPHLTTriggerMuDeDx.clone()
process.HSCPHLTTriggerMet.HLTPaths = ["HLT_MET80_*"]
process.HSCPHLTTriggerMetFilter = cms.Path(process.HSCPHLTTriggerMet   )

process.HSCPHLTTriggerPFMet = process.HSCPHLTTriggerMuDeDx.clone()
process.HSCPHLTTriggerPFMet.HLTPaths = ["HLT_PFMET150_*"]
process.HSCPHLTTriggerPFMetFilter = cms.Path(process.HSCPHLTTriggerPFMet   )

process.HSCPHLTTriggerHt = process.HSCPHLTTriggerMuDeDx.clone()
process.HSCPHLTTriggerHt.HLTPaths = ["HLT_HT650_*"]
process.HSCPHLTTriggerHtFilter = cms.Path(process.HSCPHLTTriggerHt   )

process.HSCPHLTTriggerL2Mu = process.HSCPHLTTriggerMuDeDx.clone()
process.HSCPHLTTriggerL2Mu.HLTPaths = ["HLT_L2Mu*MET*"]
process.HSCPHLTTriggerL2MuFilter = cms.Path(process.HSCPHLTTriggerL2Mu   )

process.HSCPHLTTriggerCosmic = process.HSCPHLTTriggerMuDeDx.clone()
process.HSCPHLTTriggerCosmic.HLTPaths = ["HLT_L2Mu*NoBPTX*"]
process.HSCPHLTTriggerCosmicFilter = cms.Path(process.HSCPHLTTriggerCosmic   )

if CMSSW4_2:
   #special treatment for SingleMu and PFMet trigger in 42X (2011 analysis) because threshold have changed over the year
   process.HSCPHLTTriggerPFMet = cms.EDFilter("HSCPHLTFilter",
      RemoveDuplicates = cms.bool(False),
      TriggerProcess   = cms.string("HLT"),
      MuonTrigger1Mask    = cms.int32(0),  #Activated
      PFMetTriggerMask    = cms.int32(1),  #Activated
      L2MuMETTriggerMask  = cms.int32(0),
   )
   process.HSCPHLTTriggerPFMetFilter = cms.Path(process.HSCPHLTTriggerPFMet   )

   process.HSCPHLTTriggerMu = cms.EDFilter("HSCPHLTFilter",
      RemoveDuplicates = cms.bool(False),
      TriggerProcess  = cms.string("HLT"),
      MuonTrigger1Mask    = cms.int32(1),  #Activated
      PFMetTriggerMask    = cms.int32(0),  #Activated
      L2MuMETTriggerMask  = cms.int32(0),
   )
   process.HSCPHLTTriggerMuFilter = cms.Path(process.HSCPHLTTriggerMu   )

   process.HSCPHLTTriggerL2Mu = cms.EDFilter("HSCPHLTFilter",
     RemoveDuplicates = cms.bool(False),
     TriggerProcess   = cms.string("HSCPAnalysis"),
     MuonTrigger1Mask    = cms.int32(0),  #Activated
     PFMetTriggerMask    = cms.int32(0),  #Activated
     L2MuMETTriggerMask  = cms.int32(1),  #Activated
   )

   process.HSCPHLTTriggerL2MuFilter = cms.Path(process.HSCPHLTTriggerL2Mu   )


process.Out = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep EventAux_*_*_*",
         "keep LumiSummary_*_*_*",
         "keep edmMergeableCounter_*_*_*",
         "keep *_genParticles_*_*",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep *_TrackRefitter_*_*",
         "keep *_standAloneMuons_*_*",
         "keep *_globalMuons_*_*",
         "keep *_muonsSkim_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep *_ak5PFJetsPt15__*", 
         "keep recoPFMETs_pfMet__*",     
         "keep *_HSCParticleProducer_*_*",
         "keep *_HSCPIsolation01__*",
         "keep *_HSCPIsolation03__*",
         "keep *_HSCPIsolation05__*",
         "keep *_dedx*_*_HSCPAnalysis",
         "keep *_muontiming_*_HSCPAnalysis",
         "keep triggerTriggerEvent_hltTriggerSummaryAOD_*_*",
         "keep *_RefitMTSAMuons_*_*",
         "keep *_MTMuons_*_*",
         "keep *_MTSAMuons_*_*",
         "keep *_MTmuontiming_*_*",
         "keep *_refittedStandAloneMuons_*_*",
         "keep *_offlineBeamSpot_*_*",
         "keep *_MuonSegmentProducer_*_*",
         "drop TrajectorysToOnerecoTracksAssociation_TrackRefitter__",
         "drop recoTrackExtras_*_*_*",
         "keep recoTrackExtras_TrackRefitter_*_*",
         "drop TrackingRecHitsOwned_*Muon*_*_*",
         "keep *_g4SimHits_StoppedParticles*_*",
         "keep PileupSummaryInfos_addPileupInfo_*_*"
    ),
    fileName = cms.untracked.string('XXX_OUTPUT_XXX.root'),
)

if CMSSW4_2:
   process.Out.outputCommands.extend(["keep recoPFJets_ak5PFJets__*"])

process.endPath = cms.EndPath(process.Out)
process.schedule = cms.Schedule(process.DuplicateFilter, process.HSCPHLTTriggerMuDeDxFilter, process.HSCPHLTTriggerMetDeDxFilter, process.HSCPHLTTriggerHtDeDxFilter, process.HSCPHLTTriggerMuFilter, process.HSCPHLTTriggerMetFilter, process.HSCPHLTTriggerPFMetFilter, process.HSCPHLTTriggerHtFilter, process.HSCPHLTTriggerL2MuFilter, process.HSCPHLTTriggerCosmicFilter, process.endPath)
