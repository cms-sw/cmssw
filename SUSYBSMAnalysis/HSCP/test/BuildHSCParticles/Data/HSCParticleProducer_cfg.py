import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

from SUSYBSMAnalysis.HSCP.HSCPVersion_cff import *

process.load("FWCore.MessageService.MessageLogger_cfi")
if CMSSW4_2 or CMSSW4_4:process.load("Configuration.StandardSequences.Geometry_cff")
else:                   process.load("Configuration.Geometry.GeometryIdeal_cff")

process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(100) )

if   CMSSW4_4: process.GlobalTag.globaltag = 'FT_R_44_V11::All'
elif CMSSW4_2: process.GlobalTag.globaltag = 'GR_P_V14::All'
else:          
               import FWCore.ParameterSet.VarParsing as VarParsing
               options = VarParsing.VarParsing("analysis")
               options.register("globalTag",
                   "GR_P_V32::All", # default value
                   VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                   VarParsing.VarParsing.varType.string,         # string, int, or float
                   "Global tag to be used."
               )   
               # get and parse the command line arguments
               options.parseArguments()
               process.GlobalTag.globaltag = options.globalTag

readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",
   fileNames = readFiles
)

if CMSSW4_2:   readFiles.extend(['/store/data/Run2011B/SingleMu/USER/EXOHSCP-PromptSkim-v1/0000/FC298F26-65FF-E011-977F-00237DA13C76.root'])
else:          readFiles.extend(['/store/data/Run2012D/SingleMu/USER/EXOHSCP-PromptSkim-v1/000/208/391/00000/78225FEA-B23E-E211-B4DE-485B39800C17.root'])

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

########################################################################
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM

if CMSSW4_2:
   process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
   process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi')

else:
   process.load('Configuration.Skimming.PDWG_EXOHSCP_cff')

######################################################################## INCREASING HSCP TRIGGER TRESHOLD FOR OLD DATA
process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
if CMSSW4_2 or CMSSW4_4:
   process.HSCPTrigger = cms.EDFilter("HSCPHLTFilter",
     RemoveDuplicates = cms.bool(False),
     TriggerProcess   = cms.string("HLT"),
     MuonTrigger1Mask    = cms.int32(1),  #Activated
     PFMetTriggerMask    = cms.int32(1),  #Activated
     L2MuMETTriggerMask  = cms.int32(1),  #Activated
   )
else:
   process.HSCPTrigger = process.hltHighLevel.clone()
   process.HSCPTrigger.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
   process.HSCPTrigger.HLTPaths = [
     "HLT_*_dEdx*",
     "HLT_Mu40_eta2p1*",
     "HLT_Mu50_eta2p1*",
     "HLT_HT650_*",
     "HLT_MET80_*",
     "HLT_L2Mu*MET*",
     "HLT_L2Mu*NoBPTX*",
     "HLT_PFMET150_*",
   ]
   process.HSCPTrigger.andOr = cms.bool( True ) #OR
   process.HSCPTrigger.throw = cms.bool( False )

########################################################################  SPECIAL CASE FOR DATA
process.GlobalTag.toGet = cms.VPSet(
   cms.PSet( record = cms.string('SiStripDeDxMip_3D_Rcd'),
            tag = cms.string('Data7TeV_Deco_3D_Rcd_38X'),
            connect = cms.untracked.string("sqlite_file:Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.db")),
)

if not CMSSW4_2 and not CMSSW4_4:
   print ("WARNING: You are using Data7TeV_Deco_SiStripDeDxMip_3D_Rcd.db for dEdx computation... These constants are a priori not valid for 2012 samples\nThe constants need to be redone for 2012 samples")


########################################################################
process.nEventsBefSkim  = cms.EDProducer("EventCountProducer")
process.nEventsBefEDM   = cms.EDProducer("EventCountProducer")
########################################################################

if not CMSSW4_2 and not CMSSW4_4:
   #bug fix in 52
   process.HSCParticleProducer.useBetaFromEcal = cms.bool(False)

   #skim the jet collection to keep only 15GeV jets
   process.ak5PFJetsPt15 = cms.EDFilter( "EtMinPFJetSelector",
     src = cms.InputTag( "ak5PFJets" ),
     filter = cms.bool( False ),
     etMin = cms.double( 15.0 )
   )

process.Out = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         'keep EventAux_*_*_*',
         'keep LumiSummary_*_*_*',
         'keep edmMergeableCounter_*_*_*',
         "keep *_genParticles_*_*",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep *_TrackRefitter_*_*",
         "keep *_standAloneMuons_*_*",
         "keep *_globalMuons_*_*",  #
         "keep *_muonsSkim_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep *_ak5PFJetsPt15__*", #
         "keep recoPFMETs_pfMet__*",     #
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
    ),
    fileName = cms.untracked.string('HSCP.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('p1')
    ),
)

if CMSSW4_2:
   process.Out.outputCommands.extend(["keep recoPFJets_ak5PFJets__*"])

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
#The module ak5PFJetsPt15 does not exist in CMSSW4
if CMSSW4_2 or CMSSW4_4:  process.p1 = cms.Path(process.nEventsBefSkim * process.HSCPTrigger * process.nEventsBefEDM *                         process.HSCParticleProducerSeq)
else:         process.p1 = cms.Path(process.nEventsBefSkim * process.HSCPTrigger * process.nEventsBefEDM * process.ak5PFJetsPt15 * process.HSCParticleProducerSeq)

#If you are not running from the HSCP skim you need to redo the skim
#process.p1 = cms.Path(process.nEventsBefSkim * process.HSCPTrigger * process.exoticaHSCPSeq * process.nEventsBefEDM * process.ak5PFJetsPt15 * process.HSCParticleProducerSeq)

process.endPath1 = cms.EndPath(process.Out)
process.schedule = cms.Schedule( process.p1, process.endPath1)


