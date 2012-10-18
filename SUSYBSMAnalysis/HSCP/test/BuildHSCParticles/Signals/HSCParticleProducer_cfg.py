import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.load('Configuration.StandardSequences.Services_cff')

process.options   = cms.untracked.PSet(
      wantSummary = cms.untracked.bool(True),
      SkipEvent = cms.untracked.vstring('ProductNotFound'),
)

process.MessageLogger.cerr.FwkReport.reportEvery = 1000
from SUSYBSMAnalysis.HSCP.HSCPVersion_cff import *

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(250) )

if CMSSW4_2:  process.GlobalTag.globaltag = 'START42_V9::All'
else:         process.GlobalTag.globaltag = 'START53_V7A::All'


readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",
   fileNames = readFiles
)

if CMSSW4_2: 
        readFiles.extend([
             #'/store/mc/Summer11/HSCPstau_M-308_7TeV-pythia6/GEN-SIM-RECO/PU_S4_START42_V11-v1/0000/00FA95F3-48A1-E011-9EC4-003048F0E828.root'
             '/store/user/farrell3/NewEDMFileFormat/Gluino_7TeV_M800_BX1/HSCP_1_1_OHm.root'
        ])
else:   readFiles.extend(['/store/mc/Summer12_DR53X/DYToMuMu_M_20_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/0000/5AFDB121-F6E1-E111-83D9-0018F3D0960C.root'])

process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

########################################################################

process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM


if CMSSW4_2:
   process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
   process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi')
   process.generalTracksSkim.filter       = cms.bool(False)
   process.HSCParticleProducer.filter     = cms.bool(False)
   process.DedxFilter.filter              = cms.bool(False)

else:
   process.load('Configuration.Skimming.PDWG_EXOHSCP_cff')
   process.HSCPTrigger.HLTPaths = ["*"] #not apply any trigger filter for MC
   process.HSCParticleProducer.useBetaFromEcal = cms.bool(False)
   process.HSCPEventFilter.filter = cms.bool(False)

   #skim the jet collection to keep only 15GeV jets
   process.ak5PFJetsPt15 = cms.EDFilter( "EtMinPFJetSelector",
        src = cms.InputTag( "ak5PFJets" ),
        filter = cms.bool( False ),
        etMin = cms.double( 15.0 )
   )


########################################################################  SPECIAL CASE FOR MC

process.load("PhysicsTools.HepMCCandAlgos.genParticles_cfi")
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
process.genParticles.abortOnUnknownPDGCode = cms.untracked.bool(False)

process.GlobalTag.toGet = cms.VPSet(
   cms.PSet( record = cms.string('SiStripDeDxMip_3D_Rcd'),
            tag = cms.string('MC7TeV_Deco_3D_Rcd_38X'),
            connect = cms.untracked.string("sqlite_file:MC7TeV_Deco_SiStripDeDxMip_3D_Rcd.db")),
)

process.dedxHarm2.calibrationPath      = cms.string("file:MC7TeVGains.root")
process.dedxTru40.calibrationPath      = cms.string("file:MC7TeVGains.root")
process.dedxProd.calibrationPath       = cms.string("file:MC7TeVGains.root")
process.dedxASmi.calibrationPath       = cms.string("file:MC7TeVGains.root")
process.dedxNPHarm2.calibrationPath    = cms.string("file:MC7TeVGains.root")
process.dedxNPTru40.calibrationPath    = cms.string("file:MC7TeVGains.root")
process.dedxNSHarm2.calibrationPath    = cms.string("file:MC7TeVGains.root")
process.dedxNSTru40.calibrationPath    = cms.string("file:MC7TeVGains.root")
process.dedxNPProd.calibrationPath     = cms.string("file:MC7TeVGains.root")
process.dedxNPASmi.calibrationPath     = cms.string("file:MC7TeVGains.root")
process.dedxHitInfo.calibrationPath    = cms.string("file:MC7TeVGains.root")

process.dedxHarm2.UseCalibration       = cms.bool(True)
process.dedxTru40.UseCalibration       = cms.bool(True)
process.dedxProd.UseCalibration        = cms.bool(True)
process.dedxASmi.UseCalibration        = cms.bool(True)
process.dedxNPHarm2.UseCalibration     = cms.bool(True)
process.dedxNPTru40.UseCalibration     = cms.bool(True)
process.dedxNSHarm2.UseCalibration     = cms.bool(True)
process.dedxNSTru40.UseCalibration     = cms.bool(True)
process.dedxNPProd.UseCalibration      = cms.bool(True)
process.dedxNPASmi.UseCalibration      = cms.bool(True)
process.dedxHitInfo.UseCalibration     = cms.bool(True)

if not CMSSW4_2:
   print ("WARNING: You are using MC7TeV_Deco_3D_Rcd_38X and MC7TeVGains.root for dEdx computation... These constants are a priori not valid for 2012 MC samples\nThe constants need to be redone for 2012 samples")

########################################################################
process.nEventsBefSkim  = cms.EDProducer("EventCountProducer")
process.nEventsBefEDM   = cms.EDProducer("EventCountProducer")
########################################################################

if CMSSW4_2:
   #Rerunning muon only	trigger as it did not exist in signal MC in 2011, need to modify the L1 seed to be the one available in MC
   process.load('HLTrigger.Configuration.HLT_GRun_cff')
   process.hltL1sMu16Eta2p1.L1SeedsLogicalExpression = cms.string( "L1_SingleMu16" )

process.Out = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep EventAux_*_*_*",
         "keep LumiSummary_*_*_*",
         "keep edmMergeableCounter_*_*_*",
         "keep *_genParticles_*_HSCPAnalysis",
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
         "keep *_g4SimHits_StoppedParticles*_*",
         "keep PileupSummaryInfos_addPileupInfo_*_*"
    ),
    fileName = cms.untracked.string('HSCP.root'),
#    SelectEvents = cms.untracked.PSet(
#       SelectEvents = cms.vstring('p1')
#    ),
)

if CMSSW4_2:
   process.Out.outputCommands.extend(["keep recoPFJets_ak5PFJets__*"])


########################################################################

#LOOK AT SD PASSED PATH IN ORDER to avoid as much as possible duplicated events (make the merging of .root file faster)
#The module ak5PFJetsPt15 does not exist in CMSSW4
if CMSSW4_2: process.p1 = cms.Path(process.nEventsBefSkim + process.genParticles + process.exoticaHSCPSeq + process.nEventsBefEDM                         + process.HSCParticleProducerSeq)
else:        process.p1 = cms.Path(process.nEventsBefSkim + process.genParticles + process.exoticaHSCPSeq + process.nEventsBefEDM + process.ak5PFJetsPt15 + process.HSCParticleProducerSeq)
print "You are going to run the following sequence: " + str(process.p1)

#process.p1 = cms.Path(process.HSCParticleProducerSeq)
process.endPath1 = cms.EndPath(process.Out)

if CMSSW4_2:
      #Rerun Muon only trigger for 4_2 samples
      process.schedule = cms.Schedule(process.p1, process.HLT_L2Mu60_1Hit_MET60_v6, process.endPath1)
else:
      process.schedule = cms.Schedule(process.p1, process.endPath1)
