import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.Geometry.GeometryIdeal_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )
process.MessageLogger.cerr.FwkReport.reportEvery = 1000
from SUSYBSMAnalysis.HSCP.HSCPVersion_cff import *

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(25000) )

if CMSSW4_2:  process.GlobalTag.globaltag = 'START42_V9::All'
else:         process.GlobalTag.globaltag = 'START53_V7A::All'


readFiles = cms.untracked.vstring()
process.source = cms.Source("PoolSource",
   fileNames = readFiles
)

if CMSSW4_2: readFiles.extend(['/store/user/quertenmont/11_07_30_ExoticaMCSkim//QCD_1400to1800/querten/QCD_Pt-1400to1800_TuneZ2_7TeV_pythia6/EXOHSCPMCSkim_V4_QCD_1400to1800/42f0c8f1e4a9169b4429628ad9032dfb/EXOHSCP_103_1_aV8.root'])
else:        readFiles.extend(['/store/mc/Summer12_DR53X/WToMuNu_TuneZ2star_8TeV_pythia6/GEN-SIM-RECO/PU_S10_START53_V7A-v1/0001/FE638700-BDE3-E111-A928-002618943906.root'])



process.source.inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")

########################################################################

process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM


process.load('HLTrigger.HLTfilters.hltHighLevel_cfi')
if CMSSW4_2:
   process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
   process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_EventContent_cfi')

   process.HSCPTrigger = cms.EDFilter("HSCPHLTFilter",
     RemoveDuplicates = cms.bool(False),
     TriggerProcess   = cms.string("HLT"),
     MuonTrigger1Mask    = cms.int32(1),  #Activated
     PFMetTriggerMask    = cms.int32(1),  #Activated
     L2MuMETTriggerMask  = cms.int32(1),  #Activated
   )
else:
   process.load('Configuration.Skimming.PDWG_EXOHSCP_cff')
   process.HSCParticleProducer.useBetaFromEcal = cms.bool(False)

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
process.nEventsBefEDM   = cms.EDProducer("EventCountProducer")
process.nEventsBefSkim  = cms.EDProducer("EventCountProducer")
########################################################################

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
       SelectEvents = cms.vstring('p1')
    ),
)

if CMSSW4_2:
   process.Out.outputCommands.extend(["keep recoPFJets_ak5PFJets__*"])


########################################################################

#LOOK AT SD PASSED PATH IN ORDER to avoid as much as possible duplicated events (make the merging of .root file faster)
#The module ak5PFJetsPt15 does not exist in CMSSW4
if CMSSW4_2: process.p1 = cms.Path(process.nEventsBefEDM + process.HSCPTrigger + process.HSCParticleProducerSeq)
#else:        process.p1 = cms.Path(process.nEventsBefEDM + process.HSCPTrigger + process.ak5PFJetsPt15 + process.HSCParticleProducerSeq)
else:        process.p1 = cms.Path(process.nEventsBefSkim * process.HSCPTrigger * process.exoticaHSCPSeq * process.nEventsBefEDM * process.ak5PFJetsPt15 * process.HSCParticleProducerSeq)

print "You are going to run the following sequence: " + str(process.p1)

#process.p1 = cms.Path(process.HSCParticleProducerSeq)
process.endPath1 = cms.EndPath(process.Out)
process.schedule = cms.Schedule( process.p1, process.endPath1)


