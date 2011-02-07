import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.GlobalTag.globaltag = 'GR_R_38X_V14::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
#      'rfio:/castor/cern.ch/user/q/querten/CRAB_STAGEOUT/10_12_14_HSCPSkim/V5/querten/Mu/EXOHSCPSkim7TeV/0bc44962c8c6b23d45ce69c867f520ea/EXOHSCP_91_1_zAl.root'
       '/store/group/exotica/HSCPBGSkimMu2010RunA/Mu/EXOHSCPBGSkimMu2010RunA-JC/0bc44962c8c6b23d45ce69c867f520ea/EXOHSCP_9_1_AHo.root'  #RUN136088
#        '/store/user/quertenmont/11_01_06_HSCPSkim/MuB_V2/querten/Mu/EXOHSCPSkim7TeV_V2/0bc44962c8c6b23d45ce69c867f520ea/EXOHSCP_12_1_g2S.root'
#         '/store/user/quertenmont/11_01_06_HSCPSkim/MuB_V2/querten/Mu/EXOHSCPSkim7TeV_V2/0bc44962c8c6b23d45ce69c867f520ea/EXOHSCP_87_1_8jU.root' #RUN149442
   )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")



########################################################################
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")

######################################################################## INCREASING HSCP TRIGGER TRESHOLD FOR OLD DATA

process.HSCPHLTFilter = cms.EDFilter("HSCPHLTFilter",
   TriggerProcess  = cms.string("HLT"),
   SkipJetTriggers = cms.bool(False),
)

########################################################################  SPECIAL CASE FOR DATA

process.GlobalTag.toGet = cms.VPSet(
   cms.PSet( record = cms.string('SiStripDeDxMip_3D_Rcd'),
            tag = cms.string('Data7TeV_Deco_3D_Rcd_38X'),
            connect = cms.untracked.string("sqlite_file:Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.db")),

   cms.PSet(record = cms.string("DTTtrigRcd"),
            tag = cms.string("ttrig"),
            connect = cms.untracked.string("sqlite_file:Data_v10.db")),

   cms.PSet(record = cms.string("DTMtimeRcd"),
            tag = cms.string("vdrift"),
            connect = cms.untracked.string("sqlite_file:vdrift_v9_fix_vdrift_statByView.db"))
)


process.load("RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_LinearDriftFromDBLoose_cfi")
process.dt4DSegmentsMT = process.dt4DSegments.clone()

process.muontiming.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = "dt4DSegmentsMT"
process.muontiming.TimingFillerParameters.DTTimingParameters.HitsMin = 5
process.muontiming.TimingFillerParameters.DTTimingParameters.DoWireCorr = True


########################################################################



process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep *_genParticles_*_*",
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
process.p1 = cms.Path(process.HSCPHLTFilter * process.dt4DSegmentsMT * process.HSCParticleProducerSeq)
#process.p1 = cms.Path(process.HSCParticleProducerSeq)
process.endPath = cms.EndPath(process.OUT)


