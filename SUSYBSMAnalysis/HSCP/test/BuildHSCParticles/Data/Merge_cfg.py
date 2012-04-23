import FWCore.ParameterSet.Config as cms
process = cms.Process("MergeHLT")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 5000
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
XXX_INPUT_XXX
   )
)

process.HSCPHLTDuplicate = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(True),
   TriggerProcess   = cms.string("HLT"),
   MuonTrigger1Mask    = cms.int32(0),  #Activated
   PFMetTriggerMask    = cms.int32(0),  #Activated
)
process.DuplicateFilter = cms.Path(process.HSCPHLTDuplicate   )

process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.HSCPHLTTriggerMuDeDx = process.hltHighLevel.clone()
process.HSCPHLTTriggerMuDeDx.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
process.HSCPHLTTriggerMuDeDx.andOr = cms.bool( True ) #OR
process.HSCPHLTTriggerMuDeDx.throw = cms.bool( False )
process.HSCPHLTTriggerMuDeDx.HLTPaths = ["HLT_Mu*_dEdx*"]
process.HSCPHLTTriggerMuDeDxFilter = cms.Path(process.HSCPHLTTriggerMuDeDx   )

process.HSCPHLTTriggerMetDeDx = process.HSCPHLTTriggerMuDeDx.Clone() 
process.HSCPHLTTriggerMetDeDx.HLTPaths = ["HLT_MET*_dEdx*"]
process.HSCPHLTTriggerMetDeDxFilter = cms.Path(process.HSCPHLTTriggerMetDeDx   )

process.HSCPHLTTriggerHtDeDx = process.HSCPHLTTriggerMuDeDx.Clone()
process.HSCPHLTTriggerHtDeDx.HLTPaths = ["HLT_HT*_dEdx*"]
process.HSCPHLTTriggerHtDeDxFilter = cms.Path(process.HSCPHLTTriggerHtDeDx   )

process.HSCPHLTTriggerMu = process.HSCPHLTTriggerMuDeDx.Clone()
process.HSCPHLTTriggerMu.HLTPaths = ["HLT_Mu40_*"]
process.HSCPHLTTriggerMuFilter = cms.Path(process.HSCPHLTTriggerMu   )

process.HSCPHLTTriggerMet = process.HSCPHLTTriggerMuDeDx.Clone()
process.HSCPHLTTriggerMet.HLTPaths = ["HLT_MET80_*"]
process.HSCPHLTTriggerMetFilter = cms.Path(process.HSCPHLTTriggerMet   )

process.HSCPHLTTriggerHt = process.HSCPHLTTriggerMuDeDx.Clone()
process.HSCPHLTTriggerHt.HLTPaths = ["HLT_HT650_*"]
process.HSCPHLTTriggerHtFilter = cms.Path(process.HSCPHLTTriggerHt   )

process.HSCPHLTTriggerL2Mu = process.HSCPHLTTriggerMuDeDx.Clone()
process.HSCPHLTTriggerL2Mu.HLTPaths = ["HLT_L2Mu70_eta2p1_PFMET65", "HLT_L2Mu80_eta2p1_PFMET70"]
process.HSCPHLTTriggerL2MuFilter = cms.Path(process.HSCPHLTTriggerL2Mu   )


process.Out = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         'keep EventAux_*_*_*',
         'keep LumiSummary_*_*_*',
         'keep edmMergeableCounter_*_*_*',
         "keep *_genParticles_*_*",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         #"keep *_cscSegments_*_*",
         #"keep *_rpcRecHits_*_*",
         #"keep *_dt4DSegments_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         #"keep *_reducedHSCPhbhereco_*_*",      #
         #"keep *_reducedHSCPEcalRecHitsEB_*_*", #
         #"keep *_reducedHSCPEcalRecHitsEE_*_*", #
         "keep *_TrackRefitter_*_*",
         "drop TrajectorysToOnerecoTracksAssociation_TrackRefitter__",
         "keep *_standAloneMuons_*_*",
         #"drop recoTracks_standAloneMuons__*",
         "keep *_globalMuons_*_*",  #
         "keep *_muonsSkim_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep recoPFJets_ak5PFJets__*", #
         "keep recoPFMETs_pfMet__*",     #
         "keep *_HSCParticleProducer_*_*",
         "keep *_HSCPIsolation01__*",
         "keep *_HSCPIsolation03__*",
         "keep *_HSCPIsolation05__*",
         "keep *_dedx*_*_HSCPAnalysis",
         "keep *_muontiming_*_HSCPAnalysis",
         "keep triggerTriggerEvent_hltTriggerSummaryAOD_*_*",
    ),
    fileName = cms.untracked.string('/uscmst1b_scratch/lpc1/3DayLifetime/farrell/NewDTError/XXX_OUTPUT_XXX.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('DuplicateFilter')
    ),
)

process.endPath = cms.EndPath(process.Out)

process.schedule = cms.Schedule(process.Filter, process.HSCPHLTTriggerMuDeDxFilter, process.HSCPHLTTriggerMetDeDxFilter, process.HSCPHLTTriggerHtDeDxFilter, process.HSCPHLTTriggerMuFilter, process.HSCPHLTTriggerMetFilter, process.HSCPHLTTriggerHtFilter, process.HSCPHLTTriggerL2MuFilter, process.endPath)
