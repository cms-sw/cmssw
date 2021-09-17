import FWCore.ParameterSet.Config as cms

process = cms.Process("EBTPGTest")

process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.MessageLogger.EBPhaseIITPStudies = dict()
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
   reportEvery = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2000) )

process.source = cms.Source("PoolSource",


 fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_9_0_0_pre4/RelValSingleElectronPt35_UP15/GEN-SIM-RECO/90X_mcRun2_asymptotic_v1-v1/10000/C09AD137-73EA-E611-A5F2-0CC47A4D769A.root',
        '/store/relval/CMSSW_9_0_0_pre4/RelValSingleElectronPt35_UP15/GEN-SIM-RECO/90X_mcRun2_asymptotic_v1-v1/10000/FE7D0259-73EA-E611-B051-0CC47A4D76B2.root'
),

secondaryFileNames= cms.untracked.vstring(
        '/store/relval/CMSSW_9_0_0_pre4/RelValSingleElectronPt35_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v1-v1/10000/A2D41E53-6CEA-E611-9910-0025905A60B0.root',
        '/store/relval/CMSSW_9_0_0_pre4/RelValSingleElectronPt35_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v1-v1/10000/E6142211-6DEA-E611-BDD6-0CC47A4D7658.root',
        '/store/relval/CMSSW_9_0_0_pre4/RelValSingleElectronPt35_UP15/GEN-SIM-DIGI-RAW-HLTDEBUG/90X_mcRun2_asymptotic_v1-v1/10000/E828680A-6DEA-E611-9CB6-0CC47A78A418.root')


# this is the good one 
# two files does not work here 
#fileNames = cms.untracked.vstring(
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre9/RelValSingleElectronPt35Extended/GEN-SIM-RECO/81X_mcRun2_asymptotic_v2_2023LReco-v1/10000/421F5CDF-4F53-E611-B5C2-0CC47A4D7686.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre9/RelValSingleElectronPt35Extended/GEN-SIM-RECO/81X_mcRun2_asymptotic_v2_2023LReco-v1/10000/72FC7A8C-4E53-E611-95D6-0CC47A78A360.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre9/RelValSingleElectronPt35Extended/GEN-SIM-RECO/81X_mcRun2_asymptotic_v2_2023LReco-v1/10000/BACF3CE0-4F53-E611-84E4-0025905A607E.root'
#),


#  secondaryFileNames = cms.untracked.vstring(
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre9/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/81X_mcRun2_asymptotic_v2_2023LReco-v1/10000/0E8B87C3-4953-E611-A003-0CC47A4D762A.root'
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre9/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/81X_mcRun2_asymptotic_v2_2023LReco-v1/10000/DED7A979-4A53-E611-B937-0CC47A4D762A.root'#
#)


# PU 140 
#fileNames = cms.untracked.vstring(
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/0ADFD7B5-4277-E6#11-8E89-0025905A6132.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/0C8DF598-1977-E6#11-95D4-0025905A612E.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/148A8242-1577-E6#11-B3B6-0025905B855E.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/149D7C35-1577-E6#11-992B-0CC47A745282.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/1874030D-DC76-E6#11-8FA7-0CC47A4C8E14.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/1C51682A-DD76-E6#11-A258-0025905B857A.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/1E7C6756-1677-E6#11-9633-0025905B8560.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/2007B6E9-CA76-E6#11-9B4C-0CC47A745294.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/22926245-C276-E6#11-9F2A-0025905A48B2.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/24EE0F03-D676-E6#11-B94D-0CC47A4C8E46.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/286DC4E9-CB76-E6#11-B41B-0CC47A7452D8.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/28CF1A1C-1877-E6#11-B14F-0025905A60D2.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/2C32E900-E076-E6#11-BA96-0CC47A4C8EA8.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/32FA5232-0D77-E6#11-A420-0025905B861C.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/34E99ED5-1877-E6#11-B092-0CC47A7C3572.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/40AC3F5E-F376-E6#11-BAA2-0CC47A4D76D0.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/46C79DBE-1D77-E6#11-BA00-0025905B8596.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/487702E0-0377-E6#11-B276-0025905B8560.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/48B2905D-EA76-E6#11-977F-0025905B85DC.root',
#'file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre11/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/PU25ns_81X_mcRun2_asymptotic_v5_2023D1PU140-v1/00000/48BA4FF7-D976-E6#11-A309-0CC47A4C8F26.root'
#)

# Vladimir's file 
#fileNames = cms.untracked.vstring('/store/group/dpg_trigger/comm_trigger/L1Trigger/rekovic/HGCAL/8_2_0/step2_ZEE_100ev_PU200_CMSSW_8_2_0_MinEnergy_0.5_GeV_editedSimCalorimetryEventContent_simEcalUnsuppressedDigis.root')


)



# All this stuff just runs the various EG algorithms that we are studying
                         
# ---- Global Tag :
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'PH2_1K_FB_V3::All', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')

# Choose a 2030 geometry!
#process.load('Configuration.Geometry.GeometryExtended2023simReco_cff') # Has CaloTopology, but no ECal endcap, don't use!
## Not existing in cmssw_8_1_0_pre16 process.load('Configuration.Geometry.GeometryExtended2023GRecoReco_cff') # using this geometry because I'm not sure if the tilted geometry is vetted yet
#process.load('Configuration.Geometry.GeometryExtended2023tiltedReco_cff') # this one good?

process.load('Configuration.Geometry.GeometryExtended2023D4Reco_cff')

#process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
#XXX process.load('Configuration.StandardSequences.L1TrackTrigger_cff')
#XXX process.load('Geometry.TrackerGeometryBuilder.StackedTrackerGeometry_cfi')
#XXX process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC_cfi')
#XXX process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC_cfi')

#XXX process.load('Configuration/StandardSequences/L1HwVal_cff')
#XXX process.load('Configuration.StandardSequences.RawToDigi_cff')
#XXX #XXX process.load("SLHCUpgradeSimulations.L1CaloTrigger.SLHCCaloTrigger_cff")
#XXX 
#XXX 
#XXX process.slhccalo = cms.Path( process.RawToDigi)
#XXX 
#XXX 
#XXX # run L1Reco to produce the L1EG objects corresponding
#XXX # to the current trigger
#XXX process.load('Configuration.StandardSequences.L1Reco_cff')
#XXX process.L1Reco = cms.Path( process.l1extraParticles )
#XXX 
#XXX 
#XXX 
#XXX # --------------------------------------------------------------------------------------------
#XXX #
#XXX # ----    Produce the L1EGCrystal clusters (code of Sasha Savin)
#XXX 
#XXX # first you need the ECAL RecHIts :
#XXX process.load('Configuration.StandardSequences.Reconstruction_cff')
#XXX #process.bunchSpacingProducer = cms.EDProducer("BunchSpacingProducer")
#XXX #process.bsProd = cms.Path( process.bunchSpacingProducer )
#XXX #process.reconstruction_step = cms.Path( process.bunchSpacingProducer + process.hbheprereco + process.calolocalreco )
#XXX process.reconstruction_step = cms.Path( process.bunchSpacingProducer + process.hbheUpgradeReco + process.calolocalreco )



process.simEcalEBTriggerPrimitiveDigis = cms.EDProducer("EcalEBTrigPrimProducer",
    BarrelOnly = cms.bool(True),
#    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","","HLT"),
#    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","ebDigis"),
    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
#    barrelEcalDigis = cms.InputTag("selectDigi","selectedEcalEBDigiCollection"),
    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
    TcpOutput = cms.bool(False),
    Debug = cms.bool(False),
    Famos = cms.bool(False),
    nOfSamples = cms.int32(1)
)





process.pNancy = cms.Path( process.simEcalEBTriggerPrimitiveDigis )



process.Out = cms.OutputModule( "PoolOutputModule",
    fileName = cms.untracked.string( "EBTP_PhaseII_TESTDF_uncompEt_spikeflag.root" ),
    fastCloning = cms.untracked.bool( False ),
    outputCommands = cms.untracked.vstring("keep *_EcalEBTrigPrimProducer_*_*",
                                           "keep *_TriggerResults_*_*",
                                           "keep *_ecalRecHit_EcalRecHitsEB_*",
                                           "keep *_simEcalDigis_ebDigis_*",
                                           "keep *_selectDigi_selectedEcalEBDigiCollection_*",
                                           "keep *_g4SimHits_EcalHitsEB_*",
                                           "keep *_simEcalEBTriggerPrimitiveDigis_*_*")
)

process.end = cms.EndPath( process.Out )



#print process.dumpPython()
#dump_file = open("dump_file.py", "w")
#


