import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

#The following parameters need to be provided
#isSignal, isBckg, isData, isSkimmedSample, GTAG, InputFileList
#isSignal = True
#isBckg = False
#isData = False
#isSkimmedSample = False
#GTAG = 'START72_V1::All'

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Services_cff')

process.options   = cms.untracked.PSet(
      wantSummary = cms.untracked.bool(True),
      SkipEvent = cms.untracked.vstring('ProductNotFound'),
)
process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )
process.source = cms.Source("PoolSource",
   fileNames = InputFileList,
   inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
)
if(isSignal): process.source.duplicateCheckMode = cms.untracked.string("noDuplicateCheck")


#for i in range(0,25):
#   process.source.fileNames.extend(["file:/afs/cern.ch/user/q/querten/workspace/public/14_08_12_Run2HSCP/CMSSW_7_2_X_2014-08-18-0200/src/SUSYBSMAnalysis/HSCP/test/BuildHSCParticles/Signals/../../../../../SampleProd/FARM_RECO/outputs/gluino1TeV_RECO_%04i.root" % i])

process.GlobalTag.globaltag = GTAG

process.HSCPTuplePath = cms.Path()

########################################################################
#Run the Skim sequence if necessary
if(not isSkimmedSample):
   process.nEventsBefSkim  = cms.EDProducer("EventCountProducer")

   process.load('Configuration.Skimming.PDWG_EXOHSCP_cff')
   process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
   process.HSCPTrigger = process.hltHighLevel.clone()
   process.HSCPTrigger.TriggerResultsTag = cms.InputTag( "TriggerResults", "", "HLT" )
   process.HSCPTrigger.andOr = cms.bool( True ) #OR
   process.HSCPTrigger.throw = cms.bool( False )
   if(isData):
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
   elif(isBckg):
      #to be updated to Run2 Triggers, in the meanwhile keep all of them to study trigger efficiency
      process.HSCPTrigger.HLTPaths = ["*"]
   else:
      #do not apply trigger filter on signal
      process.HSCPTrigger.HLTPaths = ["*"]  



   
   process.HSCPTuplePath += process.nEventsBefSkim + process.HSCPTrigger + process.exoticaHSCPSeq

########################################################################

#Run the HSCP EDM-tuple Sequence on skimmed sample
process.nEventsBefEDM   = cms.EDProducer("EventCountProducer")
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff") 
process.HSCPTuplePath += process.nEventsBefEDM + process.HSCParticleProducerSeq

########################################################################  
# Only for MC samples, save skimmed genParticles

if(isSignal or isBckg):
   process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
   process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
   process.allGenParticles = cms.EDProducer("GenParticleProducer",
        saveBarCodes = cms.untracked.bool(False),
        src = cms.InputTag("VtxSmeared"),
        abortOnUnknownPDGCode = cms.untracked.bool(False)
   )
   process.genParticles = cms.EDFilter("GenParticleSelector",
        filter = cms.bool(False),
        src = cms.InputTag("allGenParticles"),
        cut = cms.string('charge != 0 & pt > 5.0'),
        stableOnly = cms.bool(True)
   )

   process.HSCPTuplePath += process.allGenParticles + process.genParticles

########################################################################

#make the pool output
process.Out = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep EventAux_*_*_*",
         "keep LumiSummary_*_*_*",
         "keep edmMergeableCounter_*_*_*",
         "keep GenRunInfoProduct_*_*_*",
         "keep *_genParticles_*_HSCPAnalysis",
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
         "drop *_offlineBeamSpot_*_HSCPAnalysis", #no need to save the BS from this process
         "keep *_MuonSegmentProducer_*_*",
         "drop TrajectorysToOnerecoTracksAssociation_TrackRefitter__",
         "drop recoTrackExtras_*_*_*",
         "keep recoTrackExtras_TrackRefitter_*_*",
         "drop TrackingRecHitsOwned_*Muon*_*_*",
         "keep *_g4SimHits_StoppedParticles*_*",
         "keep PileupSummaryInfos_addPileupInfo_*_*"
    ),
    fileName = cms.untracked.string('HSCP.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('')
    ),
)

if(isBckg or isData):
   process.Out.SelectEvents.SelectEvents =  cms.vstring('HSCPTuplePath')


########################################################################

#schedule the sequence
process.endPath1 = cms.EndPath(process.Out)
process.schedule = cms.Schedule(process.HSCPTuplePath, process.endPath1)
