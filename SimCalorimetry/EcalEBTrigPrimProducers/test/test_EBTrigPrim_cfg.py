import FWCore.ParameterSet.Config as cms

process = cms.Process("EBTPGTest")

process.load('Configuration.StandardSequences.Services_cff')
process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.EventContent.EventContent_cff')
process.MessageLogger.categories = cms.untracked.vstring('EBPhaseIITPStudies', 'FwkReport')
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
   reportEvery = cms.untracked.int32(1)
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(2000) )

process.source = cms.Source("PoolSource",
#   fileNames = cms.untracked.vstring('file:/hdfs/store/mc/TTI2023Upg14D/SingleElectronFlatPt0p2To50/GEN-SIM-DIGI-RAW/PU140bx25_PH2_1K_FB_V3-v2/00000/80EE54BB-1EE6-E311-877F-002354EF3BE0.root')
#   fileNames = cms.untracked.vstring('file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre9/RelValSingleElectronPt35Extended/GEN-SIM-DIGI-RAW/81X_mcRun2_asymptotic_v2_2023LReco-v1/10000/0E8B87C3-4953-E611-A003-0CC47A4D762A.root')
 ###### fileNames = cms.untracked.vstring('file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_1_0_pre9/RelValSingleElectronPt35Extended/GEN-SIM-RECO/81X_mcRun2_asymptotic_v2_2023LReco-v1/10000/72FC7A8C-4E53-E611-95D6-0CC47A78A360.root')

#fileNames = cms.untracked.vstring(
#'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-RECO/PU25ns_81X_mcRun2_asymptotic_v8-v1/00000/B2E44B0F-6688-E611-B87B-0025905A6132.root',
#'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-RECO/PU25ns_81X_mcRun2_asymptotic_v8-v1/00000/12C68C0A-6688-E611-BB17-0CC47A4D7694.root'
#)

fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/0833A89F-7191-E611-B4F6-0025905B85BE.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/2AD714ED-5A91-E611-9A46-0025905A48EC.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/32341174-5E91-E611-A5DA-0025905A60D6.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/38849E79-5C91-E611-924D-0025905B85EE.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/440A6F90-5B91-E611-922B-0025905B8612.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/4A9E80C8-6391-E611-81B7-0CC47A78A468.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/52F3F9D3-6391-E611-BFF8-0CC47A4D762A.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/5CBEDD6B-4A91-E611-A8EA-0025905A6134.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/749CAA4F-5D91-E611-9839-0025905B85D6.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/76D5F988-6291-E611-8DC0-0CC47A4D769A.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/8A33AC3F-4B91-E611-B88F-0025905A60B4.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/984F8CA1-4891-E611-94E8-0CC47A78A456.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/9EEDA3D4-6391-E611-92C9-0CC47A4C8E22.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/A033E3D6-5C91-E611-A9F2-0CC47A78A456.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/A8BEF2B8-5F91-E611-8538-0025905A60AA.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/DC4BF1BD-5491-E611-9968-0CC47A78A456.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/DE1C7FE0-4991-E611-8A32-0025905B8562.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/EA27CFBB-4D91-E611-BFCC-0CC47A4C8E28.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/F4E4CE9B-5691-E611-9941-0CC47A4D769A.root',
'/store/relval/CMSSW_8_1_0_pre12/RelValH125GGgluonfusion_13/GEN-SIM-DIGI-RAW/PU25ns_81X_upgrade2017_HCALdev_v2_BpixFpixHcalGeom-v1/00000/F64137EF-5E91-E611-8D7A-0CC47A4C8EE8.root'
)


   #fileNames = cms.untracked.vstring('file:root://cmsxrootd.fnal.gov///store/relval/CMSSW_8_0_5/RelValQCD_FlatPt_15_3000HS_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_2016_miniAODv2_v0_gs71xJecGT-v1/00000/06A1518F-3311-E611-BC5A-0CC47A78A496.root')
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

process.load('Configuration.Geometry.GeometryExtended2023D5Reco_cff')

#process.load('Configuration.Geometry.GeometryExtended2016Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_38T_PostLS1_cff')
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



#process.simEcalEBTriggerPrimitiveDigis = cms.EDProducer("EcalEBTrigPrimProducer",
process.EcalEBTrigPrimProducer = cms.EDProducer("EcalEBTrigPrimProducer",
    BarrelOnly = cms.bool(True),
#    barrelEcalDigis = cms.InputTag("simEcalUnsuppressedDigis","ebDigis"),
    barrelEcalDigis = cms.InputTag("simEcalDigis","ebDigis"),
#    barrelEcalDigis = cms.InputTag("selectDigi","selectedEcalEBDigiCollection"),
    binOfMaximum = cms.int32(6), ## optional from release 200 on, from 1-10
    TcpOutput = cms.bool(False),
    Debug = cms.bool(True),
    Famos = cms.bool(False),
    nOfSamples = cms.int32(1)
)




process.pNancy = cms.Path( process.EcalEBTrigPrimProducer )
#process.pNancy = cms.Path( process.simEcalEBTriggerPrimitiveDigis )



process.Out = cms.OutputModule( "PoolOutputModule",
#    fileName = cms.untracked.string( "EBTP_PhaseII_RelVal2017.root" ),
    fileName = cms.untracked.string( "EBTP_PhaseII.root" ),
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


