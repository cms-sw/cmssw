import FWCore.ParameterSet.Config as cms
process = cms.Process("MergeHLT")

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
   MuonTrigger1Mask    = cms.int32(1),  #Activated
   PFMetTriggerMask    = cms.int32(1),  #Activated
)

process.HSCPHLTFilterPFMET = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(False),
   TriggerProcess   = cms.string("HLT"),
   MuonTrigger1Mask    = cms.int32(0),  #Activated
   PFMetTriggerMask    = cms.int32(1),  #Activated
)


process.HSCPHLTFilterSingleMU = cms.EDFilter("HSCPHLTFilter",
   RemoveDuplicates = cms.bool(False),
   TriggerProcess  = cms.string("HLT"),
   MuonTrigger1Mask    = cms.int32(1),  #Activated
   PFMetTriggerMask    = cms.int32(0),  #Activated
)


process.Filter      = cms.Path(process.HSCPHLTDuplicate   )
process.HscpPathPFMet = cms.Path(process.HSCPHLTFilterPFMET   )
process.HscpPathSingleMu  = cms.Path(process.HSCPHLTFilterSingleMU    )


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
    fileName = cms.untracked.string('/uscmst1b_scratch/lpc1/3DayLifetime/farrell/UpdateTo4p3fb/XXX_OUTPUT_XXX.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('Filter')
    ),
)

process.endPath = cms.EndPath(process.Out)

process.schedule = cms.Schedule(process.Filter, process.HscpPathPFMet, process.HscpPathSingleMu, process.endPath)


