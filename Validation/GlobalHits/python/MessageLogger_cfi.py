import FWCore.ParameterSet.Config as cms

MessageLogger = cms.Service("MessageLogger",
    MessageLogger = cms.untracked.PSet(
        Root_Warning = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_Error = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EventSetupDependency = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_GlobalRecHitsProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_GlobalDigisProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        lineLength = cms.untracked.int32(132),
        GlobalRecHitsProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_GlobalHitsProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_GlobalRecHitsHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_GlobalHitsHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_GlobalDigisHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_GlobalRecHitsAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_GlobalHitsProdHistStripper = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        GlobalHitsProdHistStripper_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        GlobalHitsAnalyzer_GlobalHitsAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_GlobalDigisAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ScheduleExecutionFailure = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_GlobalHitsProdHist = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    cout = cms.untracked.PSet(
        threshold = cms.untracked.string('INFO'),
        GlobalDigisProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_GlobalRecHitsProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_GlobalDigisProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        lineLength = cms.untracked.int32(132),
        GlobalRecHitsProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_GlobalHitsProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_GlobalRecHitsHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_GlobalHitsHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_GlobalDigisHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_GlobalRecHitsAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_GlobalHitsProdHistStripper = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        GlobalHitsProdHistStripper_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        GlobalHitsAnalyzer_GlobalHitsAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_GlobalDigisAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_GlobalHitsProdHist = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    cerr = cms.untracked.PSet(
        Root_Warning = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        threshold = cms.untracked.string('WARNING'),
        GlobalDigisProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        Root_Error = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        EventSetupDependency = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_GlobalRecHitsProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_GlobalDigisProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        lineLength = cms.untracked.int32(132),
        GlobalRecHitsProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_clear = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_GlobalHitsProducer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_GlobalRecHitsHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_GlobalHitsHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_GlobalDigisHistogrammer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeG4MC = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_GlobalRecHitsAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillMuon = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_GlobalHitsProdHistStripper = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_produce = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        default = cms.untracked.PSet(
            limit = cms.untracked.int32(0)
        ),
        GlobalHitsProdHistStripper_beginRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        noLineBreaks = cms.untracked.bool(True),
        GlobalHitsAnalyzer_GlobalHitsAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHistStripper_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_GlobalDigisAnalyzer = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_endJob = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_endRun = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalRecHitsAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProducer_storeECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        ScheduleExecutionFailure = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillHCal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_GlobalHitsProdHist = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisHistogrammer_analyze = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalHitsProdHist_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisAnalyzer_fillECal = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        ),
        GlobalDigisProducer_fillTrk = cms.untracked.PSet(
            limit = cms.untracked.int32(10000000)
        )
    ),
    categories = cms.untracked.vstring( 
        'GlobalHitsProducer_GlobalHitsProducer', 
        'GlobalHitsProducer_endJob', 
        'GlobalHitsProducer_produce', 
        'GlobalHitsProducer_fillG4MC', 
        'GlobalHitsProducer_storeG4MC', 
        'GlobalHitsProducer_fillTrk', 
        'GlobalHitsProducer_storeTrk', 
        'GlobalHitsProducer_fillMuon', 
        'GlobalHitsProducer_storeMuon', 
        'GlobalHitsProducer_fillECal', 
        'GlobalHitsProducer_storeECal', 
        'GlobalHitsProducer_fillHCal', 
        'GlobalHitsProducer_storeHCal', 
        'GlobalHitsProducer_clear', 
        'GlobalDigisProducer_GlobalDigisProducer', 
        'GlobalDigisProducer_endJob', 
        'GlobalDigisProducer_produce', 
        'GlobalDigisProducer_fillG4MC', 
        'GlobalDigisProducer_storeG4MC', 
        'GlobalDigisProducer_fillTrk', 
        'GlobalDigisProducer_storeTrk', 
        'GlobalDigisProducer_fillMuon', 
        'GlobalDigisProducer_storeMuon', 
        'GlobalDigisProducer_fillECal', 
        'GlobalDigisProducer_storeECal', 
        'GlobalDigisProducer_fillHCal', 
        'GlobalDigisProducer_storeHCal', 
        'GlobalDigisProducer_clear', 
        'GlobalRecHitsProducer_GlobalRecHitsProducer', 
        'GlobalRecHitsProducer_endJob', 
        'GlobalRecHitsProducer_produce', 
        'GlobalRecHitsProducer_fillG4MC', 
        'GlobalRecHitsProducer_storeG4MC', 
        'GlobalRecHitsProducer_fillTrk', 
        'GlobalRecHitsProducer_storeTrk', 
        'GlobalRecHitsProducer_fillMuon', 
        'GlobalRecHitsProducer_storeMuon', 
        'GlobalRecHitsProducer_fillECal', 
        'GlobalRecHitsProducer_storeECal', 
        'GlobalRecHitsProducer_fillHCal', 
        'GlobalRecHitsProducer_storeHCal', 
        'GlobalRecHitsProducer_clear', 
        'ScheduleExecutionFailure', 
        'EventSetupDependency', 
        'Root_Warning', 
        'Root_Error', 
        'GlobalHitsAnalyzer_GlobalHitsAnalyzer', 
        'GlobalHitsAnalyzer_endJob', 
        'GlobalHitsAnalyzer_analyze', 
        'GlobalHitsAnalyzer_fillG4MC', 
        'GlobalHitsAnalyzer_fillTrk', 
        'GlobalHitsAnalyzer_fillMuon', 
        'GlobalHitsAnalyzer_fillECal', 
        'GlobalHitsAnalyzer_fillHCal', 
        'GlobalDigisAnalyzer_GlobalDigisAnalyzer', 
        'GlobalDigisAnalyzer_endJob', 
        'GlobalDigisAnalyzer_analyze', 
        'GlobalDigisAnalyzer_fillTrk', 
        'GlobalDigisAnalyzer_fillMuon', 
        'GlobalDigisAnalyzer_fillECal', 
        'GlobalDigisAnalyzer_fillHCal', 
        'GlobalRecHitsAnalyzer_GlobalRecHitsAnalyzer', 
        'GlobalRecHitsAnalyzer_endJob', 
        'GlobalRecHitsAnalyzer_analyze', 
        'GlobalRecHitsAnalyzer_fillTrk', 
        'GlobalRecHitsAnalyzer_fillMuon', 
        'GlobalRecHitsAnalyzer_fillECal', 
        'GlobalRecHitsAnalyzer_fillHCal', 
        'GlobalHitsHistogrammer_GlobalHitsHistogrammer', 
        'GlobalHitsHistogrammer_endJob', 
        'GlobalHitsHistogrammer_analyze', 
        'GlobalDigisHistogrammer_GlobalDigisHistogrammer', 
        'GlobalDigisHistogrammer_endJob', 
        'GlobalDigisHistogrammer_analyze', 
        'GlobalRecHitsHistogrammer_GlobalRecHitsHistogrammer', 
        'GlobalRecHitsHistogrammer_endJob', 
        'GlobalRecHitsHistogrammer_analyze', 
        'GlobalHitsProdHist_GlobalHitsProdHist', 
        'GlobalHitsProdHist_endJob', 
        'GlobalHitsProdHist_produce', 
        'GlobalHitsProdHist_fillG4MC', 
        'GlobalHitsProdHist_fillTrk', 
        'GlobalHitsProdHist_fillMuon', 
        'GlobalHitsProdHist_fillECal', 
        'GlobalHitsProdHist_fillHCal', 
        'GlobalHitsProdHist_endRun', 
        'GlobalHitsProdHistStripper_GlobalHitsProdHistStripper', 
        'GlobalHitsProdHistStripper_endJob', 
        'GlobalHitsProdHistStripper_beginRun', 
        'GlobalHitsProdHistStripper_endRun'),
    destinations = cms.untracked.vstring('MessageLogger', 
        'cout', 
        'cerr')
)


