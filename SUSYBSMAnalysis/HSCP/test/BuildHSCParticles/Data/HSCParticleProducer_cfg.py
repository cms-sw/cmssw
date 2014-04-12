import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.GlobalTag.globaltag = 'GR_P_V14::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/data/Run2011B/MET/USER/EXOHSCP-PromptSkim-v1/0000/F6FA7586-9B02-E111-B438-001A92810AEC.root'
#        '/store/data/Run2011A/SingleMu/USER/EXOHSCP-05Aug2011-v1/0000/FC150FF5-0FC5-E011-913E-002618943876.root',
#        '/store/data/Run2011A/SingleMu/USER/EXOHSCP-05Aug2011-v1/0000/FA8B74F8-0FC5-E011-9EE5-001A928116B4.root',
#        '/store/data/Run2011A/SingleMu/USER/EXOHSCP-05Aug2011-v1/0000/FA432D78-CEC5-E011-B4F3-00261894394F.root'
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
   RemoveDuplicates    = cms.bool(False),
   MuonTrigger1Mask    = cms.int32(1),  #Activated
   PFMetTriggerMask    = cms.int32(0),  #Deactivated
)

########################################################################  SPECIAL CASE FOR DATA

process.GlobalTag.toGet = cms.VPSet(
   cms.PSet( record = cms.string('SiStripDeDxMip_3D_Rcd'),
            tag = cms.string('Data7TeV_Deco_3D_Rcd_38X'),
            connect = cms.untracked.string("sqlite_file:Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.db")),
)

process.load("RecoLocalMuon.DTSegment.dt4DSegments_MTPatternReco4D_LinearDriftFromDBLoose_cfi")
process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.AlphaMaxPhi = 1.0
process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.AlphaMaxTheta = 0.9
process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.segmCleanerMode = 2
process.dt4DSegments.Reco4DAlgoConfig.Reco2DAlgoConfig.MaxChi2 = 1.0
process.dt4DSegmentsMT = process.dt4DSegments.clone()
process.dt4DSegmentsMT.Reco4DAlgoConfig.recAlgoConfig.stepTwoFromDigi = True
process.dt4DSegmentsMT.Reco4DAlgoConfig.Reco2DAlgoConfig.recAlgoConfig.stepTwoFromDigi = True

process.muontiming.TimingFillerParameters.DTTimingParameters.MatchParameters.DTsegments = "dt4DSegmentsMT"
process.muontiming.TimingFillerParameters.DTTimingParameters.HitsMin = 3
process.muontiming.TimingFillerParameters.DTTimingParameters.RequireBothProjections = False
process.muontiming.TimingFillerParameters.DTTimingParameters.DropTheta = True
process.muontiming.TimingFillerParameters.DTTimingParameters.DoWireCorr = True
process.muontiming.TimingFillerParameters.DTTimingParameters.MatchParameters.DTradius = 1.0


########################################################################
process.nEventsBefEDM   = cms.EDProducer("EventCountProducer")
########################################################################

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
         "drop recoTracks_standAloneMuons__*",
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
    fileName = cms.untracked.string('HSCP.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('p1')
    ),
)

from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup 
process.tTrigDB = cms.ESSource("PoolDBESSource",
                                   CondDBSetup,
                                   timetype = cms.string('runnumber'),
                                   toGet = cms.VPSet(cms.PSet(
                                       record = cms.string('DTTtrigRcd'),
                                       tag = cms.string('DTTtrig_offline_prep_V03'),
                                       label = cms.untracked.string('')
                                   )),
                                   connect = cms.string('frontier://FrontierPrep/CMS_COND_DT'),
                                   authenticationMethod = cms.untracked.uint32(0)
                                   )
#process.tTrigDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.es_prefer_tTrigDB = cms.ESPrefer('PoolDBESSource','tTrigDB')

process.vDriftDB = cms.ESSource("PoolDBESSource",
                                   CondDBSetup,
                                   timetype = cms.string('runnumber'),
                                   toGet = cms.VPSet(cms.PSet(
                                       record = cms.string('DTMtimeRcd'),
                                       tag = cms.string('DTVdrift_offline_prep_V03'),
                                       label = cms.untracked.string('')
                                   )),
                                   connect = cms.string('frontier://FrontierPrep/CMS_COND_DT'),
                                   authenticationMethod = cms.untracked.uint32(0)
                                   )
#process.vDriftDB.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'
process.es_prefer_vDriftDB = cms.ESPrefer('PoolDBESSource','vDriftDB')

########################################################################


#LOOK AT SD PASSED PATH IN ORDER to avoid as much as possible duplicated events (make the merging of .root file faster)
process.p1 = cms.Path(process.nEventsBefEDM * process.HSCPHLTFilter * process.dt4DSegmentsMT * process.HSCParticleProducerSeq)
#process.p1 = cms.Path(process.HSCParticleProducerSeq)
process.endPath1 = cms.EndPath(process.Out)
process.schedule = cms.Schedule( process.p1, process.endPath1)


