import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.GlobalTag.globaltag = 'GR_P_V32::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
        '/store/data/Run2012A/SingleMu/RECO/PromptReco-v1/000/191/248/186722DA-5E88-E111-ADFF-003048D2C0F0.root'
   )
)
process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
#process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('190645:10-190645:110')
#import FWCore.PythonUtilities.LumiList as LumiList
#process.source.lumisToProcess = LumiList.LumiList(filename = 'OfficialLumi.json').getVLuminosityBlockRange()

########################################################################
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")

######################################################################## INCREASING HSCP TRIGGER TRESHOLD FOR OLD DATA

process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
process.HSCPTrigger = process.hltHighLevel.clone()
process.HSCPTrigger.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
process.HSCPTrigger.HLTPaths = [
    "HLT_*_dEdx*",
    "HLT_Mu40_eta2p1*",
    "HLT_Mu50_eta2p1*",
    "HLT_HT650_*",
    "HLT_MET80_*",
    "HLT_L2Mu*_eta2p1_PFMET*",
]
process.HSCPTrigger.andOr = cms.bool( True ) #OR
process.HSCPTrigger.throw = cms.bool( False )


########################################################################  SPECIAL CASE FOR DATA

process.GlobalTag.toGet = cms.VPSet(
   cms.PSet( record = cms.string('SiStripDeDxMip_3D_Rcd'),
            tag = cms.string('Data7TeV_Deco_3D_Rcd_38X'),
            connect = cms.untracked.string("sqlite_file:Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.db")),
)

########################################################################
process.nEventsBefSkim  = cms.EDProducer("EventCountProducer")
process.nEventsBefEDM   = cms.EDProducer("EventCountProducer")
########################################################################

#bug fix in 52
process.HSCParticleProducer.useBetaFromEcal = cms.bool(False)

#no dE/dx cut from skim:
process.DedxFilter.filter = cms.bool(False)


process.Out = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         'keep EventAux_*_*_*',
         'keep LumiSummary_*_*_*',
         'keep edmMergeableCounter_*_*_*',
         "keep *_genParticles_*_*",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         "keep *_cscSegments_*_*",
         #"keep *_rpcRecHits_*_*",
         "keep *_dt4DSegments_*_*",
         "keep *_dt4DSegmentsMT_*_*",
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
         "keep *_RefitMTSAMuons_*_*",
         "keep *_RefitMTMuons_*_*",
         "keep *_MTmuontiming_*_*",
         "keep *_offlineBeamSpot_*_*",
         "keep *_MuonSegmentProducer_*_*",
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
process.p1 = cms.Path(process.nEventsBefSkim * process.HSCPTrigger * process.exoticaHSCPSeq * process.nEventsBefEDM * process.HSCParticleProducerSeq)
#process.p1 = cms.Path(process.HSCParticleProducerSeq)
process.endPath1 = cms.EndPath(process.Out)
process.schedule = cms.Schedule( process.p1, process.endPath1)


