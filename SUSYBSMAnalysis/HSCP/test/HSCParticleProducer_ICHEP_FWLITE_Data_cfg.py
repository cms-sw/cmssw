import FWCore.ParameterSet.Config as cms

process = cms.Process("HSCPAnalysis")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.GlobalTag.globaltag = 'GR_R_36X_V12A::All'

process.source = cms.Source("PoolSource",
   fileNames = cms.untracked.vstring(
   )
)



########################################################################

import HLTrigger.HLTfilters.hltHighLevelDev_cfi


### JetMETTau SD
process.JetMETTau_1e28 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.JetMETTau_1e28.HLTPaths = (
#"HLT_Jet15U",
#"HLT_DiJetAve15U_8E29",
"HLT_FwdJet20U",
"HLT_Jet30U", 
"HLT_Jet50U",
"HLT_DiJetAve30U_8E29",
"HLT_QuadJet15U",
"HLT_MET45",
"HLT_MET100",
"HLT_HT100U",
"HLT_SingleLooseIsoTau20",
"HLT_DoubleLooseIsoTau15",
"HLT_DoubleJet15U_ForwardBackward",
"HLT_BTagMu_Jet10U",
"HLT_BTagIP_Jet50U",
"HLT_StoppedHSCP_8E29"
)
process.JetMETTau_1e28.HLTPathsPrescales  = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1)
process.JetMETTau_1e28.HLTOverallPrescale = cms.uint32(1)
process.JetMETTau_1e28.throw = False
process.JetMETTau_1e28.andOr = True

### Mu SD
process.Mu_1e28 = HLTrigger.HLTfilters.hltHighLevelDev_cfi.hltHighLevelDev.clone(andOr = True)
process.Mu_1e28.HLTPaths = (
#"HLT_L2Mu0",
#"HLT_L2Mu3",
#"HLT_L2Mu5",
"HLT_L1Mu20",
"HLT_L2Mu9",
"HLT_L2Mu11",
"HLT_L1Mu14_L1SingleEG10",
"HLT_L1Mu14_L1SingleJet6U",
"HLT_L1Mu14_L1ETM30",
"HLT_L2DoubleMu0",
"HLT_L1DoubleMuOpen",
"HLT_DoubleMu0",
"HLT_DoubleMu3",
"HLT_Mu3",
"HLT_Mu5",
"HLT_Mu9",
"HLT_IsoMu3",
"HLT_Mu0_L1MuOpen",
"HLT_Mu0_Track0_Jpsi",
"HLT_Mu3_L1MuOpen",
"HLT_Mu3_Track0_Jpsi",
"HLT_Mu5_L1MuOpen",
"HLT_Mu5_Track0_Jpsi",
"HLT_Mu0_L2Mu0",
"HLT_Mu3_L2Mu0",
"HLT_Mu5_L2Mu0"
)
process.Mu_1e28.HLTPathsPrescales  = cms.vuint32(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1)
process.Mu_1e28.HLTOverallPrescale = cms.uint32(1)
process.Mu_1e28.throw = False
process.Mu_1e28.andOr = True



########################################################################
process.load('SUSYBSMAnalysis.Skimming.EXOHSCP_cff')
process.load("SUSYBSMAnalysis.HSCP.HSCParticleProducerFromSkim_cff")  #IF RUNNING ON HSCP SKIM
process.load("SUSYBSMAnalysis.HSCP.HSCPTreeBuilder_cff")

################## DEDX ANALYSIS SEQUENCE MODULES ##################

#process.TFileService = cms.Service("TFileService", 
#        fileName = cms.string('HSCP_tree.root')
#)

process.OUT = cms.OutputModule("PoolOutputModule",
     outputCommands = cms.untracked.vstring(
         "drop *",
         "keep GenEventInfoProduct_generator_*_*",
         "keep *_offlinePrimaryVertices_*_*",
         "keep *_csc2DRecHits_*_*",
         "keep *_cscSegments_*_*",
         "keep *_dt1DRecHits_*_*",
         "keep *_rpcRecHits_*_*",
         "keep *_dt4DSegments_*_*",
         "keep SiStripClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep SiPixelClusteredmNewDetSetVector_generalTracksSkim_*_*",
         "keep *_reducedHSCPhbhereco_*_*",
         "keep *_reducedHSCPEcalRecHitsEB_*_*",
         "keep *_TrackRefitter_*_*",
         "keep *_standAloneMuons_*_*",
         "keep *_globalMuons_*_*",
         "keep *_muonsSkim_*_*",
         "keep L1GlobalTriggerReadoutRecord_gtDigis_*_*",
         "keep edmTriggerResults_TriggerResults_*_*",
         "keep *_HSCParticleProducer_*_*",
    ),
    fileName = cms.untracked.string('HSCP.root'),
    SelectEvents = cms.untracked.PSet(
       SelectEvents = cms.vstring('p1','p2')
    ),
)

########################################################################


#LOOK AT SD PASSED PATH IN ORDER to avoid as much as possible duplicated events (make the merging of .root file faster)
process.p1 = cms.Path(process.Mu_1e28 * process.HSCParticleProducerSeq)
process.p2 = cms.Path(process.JetMETTau_1e28 * ~process.Mu_1e28 * process.HSCParticleProducerSeq)
process.endPath = cms.EndPath(process.OUT)
