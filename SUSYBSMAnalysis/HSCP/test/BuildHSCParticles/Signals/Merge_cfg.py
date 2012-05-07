import FWCore.ParameterSet.Config as cms
process = cms.Process("Merge")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 50000
process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
XXX_INPUT_XXX
   )
)

process.HSCPHLTDuplicate = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(True),
   TriggerProcess   = cms.string("HLT"),
   MuonTriggerMask  = cms.int32(1),  #Ignored
   METTriggerMask   = cms.int32(1),  #Ignored
   JetTriggerMask   = cms.int32(1),  #Ignored
)

process.HSCPHLTFilterMET = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(False),
   TriggerProcess   = cms.string("HLT"),
   MuonTriggerMask  = cms.int32(0),  #Ignored
   METTriggerMask   = cms.int32(1),  #Activated
   JetTriggerMask   = cms.int32(0),  #Ignored
)

process.HSCPHLTFilterMU = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(False),
   TriggerProcess  = cms.string("HLT"),
   MuonTriggerMask = cms.int32(1),  #Activated
   METTriggerMask  = cms.int32(0), #Ignored
   JetTriggerMask  = cms.int32(0),  #Ignored
)

process.HSCPHLTFilterJET = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(False),
   TriggerProcess  = cms.string("HLT"),
   MuonTriggerMask = cms.int32(0), #Ignored
   METTriggerMask  = cms.int32(0), #Ignored
   JetTriggerMask  = cms.int32(1), #Activated
)

process.Filter      = cms.Path(process.HSCPHLTDuplicate   )
process.HscpPathMet = cms.Path(process.HSCPHLTFilterMET   )
process.HscpPathMu  = cms.Path(process.HSCPHLTFilterMU    )
process.HscpPathJet = cms.Path(process.HSCPHLTFilterJET   )


process.Out = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep *_genParticles_*_*",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         "keep *_cscSegments_*_*",
         "keep *_rpcRecHits_*_*",
         "keep *_dt4DSegments_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep *_reducedHSCPhbhereco_*_*",      #
         "keep *_reducedHSCPEcalRecHitsEB_*_*", #
         "keep *_reducedHSCPEcalRecHitsEE_*_*", #
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
    fileName = cms.untracked.string('XXX_OUTPUT_XXX.root'),
)

process.endPath = cms.EndPath(process.Out)

process.schedule = cms.Schedule(process.Filter, process.HscpPathMet, process.HscpPathMu, process.HscpPathJet, process.endPath)


