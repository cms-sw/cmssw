import FWCore.ParameterSet.Config as cms
from SLHCUpgradeSimulations.Configuration.postLS1Customs import customisePostLS1
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import customise as customiseBE
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import customise as customiseBE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10D import customise as customiseBE5DPixel10D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10DLHCC import customise as customiseBE5DPixel10DLHCC
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10DLHCCCooling import customise as customiseBE5DPixel10DLHCCCooling
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10Ddev import customise as customiseBE5DPixel10Ddev

from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE import l1EventContent as customise_ev_BE
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5D import l1EventContent as customise_ev_BE5D
from SLHCUpgradeSimulations.Configuration.phase2TkCustomsBE5DPixel10D import l1EventContent as customise_ev_BE5DPixel10D

from SLHCUpgradeSimulations.Configuration.phase1TkCustomsPixel10D import customise as customisePhase1TkPixel10D
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import customise as customiseTTI
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import l1EventContent_TTI as customise_ev_l1tracker
from SLHCUpgradeSimulations.Configuration.combinedCustoms_TTI import l1EventContent_TTI_forHLT

from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_NoCrossing
from SLHCUpgradeSimulations.Configuration.phase1TkCustoms import customise as customisePhase1Tk
from SLHCUpgradeSimulations.Configuration.phase1TkCustomsdev import customise as customisePhase1Tkdev
from SLHCUpgradeSimulations.Configuration.HCalCustoms import customise_HcalPhase1, customise_HcalPhase0, customise_HcalPhase2
from SLHCUpgradeSimulations.Configuration.gemCustoms import customise2019 as customise_gem2019
from SLHCUpgradeSimulations.Configuration.gemCustoms import customise2023 as customise_gem2023
from SLHCUpgradeSimulations.Configuration.me0Customs import customise as customise_me0
from SLHCUpgradeSimulations.Configuration.rpcCustoms import customise as customise_rpc
from SLHCUpgradeSimulations.Configuration.fastsimCustoms import customiseDefault as fastCustomiseDefault
from SLHCUpgradeSimulations.Configuration.fastsimCustoms import customisePhase2 as fastCustomisePhase2
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_noPixelDataloss as cNoPixDataloss
from SLHCUpgradeSimulations.Configuration.customise_ecalTime import cust_ecalTime
from SLHCUpgradeSimulations.Configuration.customise_shashlikTime import cust_shashlikTime
import SLHCUpgradeSimulations.Configuration.aging as aging
import SLHCUpgradeSimulations.Configuration.jetCustoms as jetCustoms

def cust_phase1_Pixel10D(process):
    process=customisePostLS1(process)
    process=customisePhase1TkPixel10D(process)
    process=customise_HcalPhase1(process)
    process=jetCustoms.customise_jets(process)
    return process 

def cust_phase2_BE5DPixel10D(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_phase2_BE5DPixel10DLHCC(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10DLHCC(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_phase2_BE5DPixel10DLHCCCooling(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10DLHCCCooling(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_phase2_BE5D(process):
    process=customisePostLS1(process)
    process=customiseBE5D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5D(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_phase2_BE(process):
    process=customisePostLS1(process)
    process=customiseBE(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_2017(process):
    process=customisePostLS1(process)
    process=customisePhase1Tk(process)
    process=customise_HcalPhase0(process)
#    process=fixRPCConditions(process)
    return process

def cust_2017dev(process):
    process=customisePostLS1(process)
    process=customisePhase1Tkdev(process)
    process=customise_HcalPhase0(process)
#    process=fixRPCConditions(process)
    return process

def cust_2017EcalTime(process):
    process=cust_2017(process)
    process=cust_ecalTime(process)
    return process

def cust_2019(process):
    process=customisePostLS1(process)
    process=customisePhase1Tk(process)
    process=customise_HcalPhase1(process)
    process=jetCustoms.customise_jets(process)
#    process=fixRPCConditions(process)
    return process

def cust_2019WithGem(process):
    process=cust_2019(process)
    process=customise_gem2019(process)
    return process

def cust_2023(process):
    process=customisePostLS1(process)
    process=customiseBE5D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5D(process)
    process=customise_gem2023(process)
    process=customise_rpc(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_2023SHCal(process):
    process=cust_2023Muon(process)
    if hasattr(process,'L1simulation_step'):
        process.simEcalTriggerPrimitiveDigis.BarrelOnly = cms.bool(True)
    if hasattr(process,'digitisation_step'):
        process.mix.digitizers.ecal.accumulatorType = cms.string('EcalPhaseIIDigiProducer')
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits","EcalHitsEK") )
        process.mix.mixObjects.mixCH.subdets.append( "EcalHitsEK" )
        process.simEcalUnsuppressedDigis = cms.EDAlias(
            mix = cms.VPSet(
            cms.PSet(type = cms.string('EBDigiCollection')),
            cms.PSet(type = cms.string('EEDigiCollection')),
            cms.PSet(type = cms.string('EKDigiCollection')),
            cms.PSet(type = cms.string('ESDigiCollection'))
            )
            )
        
    if hasattr(process,'raw2digi_step'):
        process.ecalDigis.FEDs = cms.vint32(
            # EE-:
            #601, 602, 603, 604, 605,
            #606, 607, 608, 609,
            # EB-:
            610, 611, 612, 613, 614, 615,
            616, 617, 618, 619, 620, 621,
            622, 623, 624, 625, 626, 627,
            # EB+:
            628, 629, 630, 631, 632, 633,
            634, 635, 636, 637, 638, 639,
            640, 641, 642, 643, 644, 645,
            # EE+:
            #646, 647, 648, 649, 650,
            #651, 652, 653, 654
            )
        print "RAW2DIGI only for EB FEDs"

    if hasattr(process,'reconstruction_step'):
    	process.ecalRecHit.EEuncalibRecHitCollection = cms.InputTag("","")
        process.reducedEcalRecHitsSequence.remove(process.reducedEcalRecHitsES)
        #remove the old EE pfrechit producer
        del process.particleFlowRecHitECAL.producers[1]
        process.particleFlowClusterEBEKMerger = cms.EDProducer('PFClusterCollectionMerger',
                                                               inputs = cms.VInputTag(cms.InputTag('particleFlowClusterECALUncorrected'),
                                                                                      cms.InputTag('particleFlowClusterEKUncorrected')
                                                                                      )
                                                               )   
        process.pfClusteringECAL.remove(process.particleFlowClusterECAL)
#        process.pfClusteringECAL.remove(process.particleFlowClusterECALWithTimeSelected)
        process.pfClusteringECAL += process.pfClusteringEK 
        process.pfClusteringECAL += process.particleFlowClusterEBEKMerger
        process.pfClusteringECAL += process.particleFlowClusterECAL        
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterEBEKMerger')
        process.pfClusteringECAL += process.particleFlowClusterECAL
        #process.particleFlowCluster += process.pfClusteringEK
        
        #clone photons to mustache photons so we can compare back to old reco
        process.mustachePhotonCore = process.photonCore.clone(scHybridBarrelProducer = cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),scIslandEndcapProducer = cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"))
        process.mustachePhotons = process.photons.clone(photonCoreProducer = cms.InputTag('mustachePhotonCore'), endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEK"))        
        process.photonSequence += process.mustachePhotonCore
        process.photonSequence += process.mustachePhotons
        #point particle flow at the right photon collection     
        process.particleFlowBlock.elementImporters[2].source = cms.InputTag('mustachePhotons')
        process.gedPhotons.endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        
        process.towerMaker.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEK"))
        process.towerMakerPF.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEK"))
        process.towerMakerWithHO.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEK"))
        process.towerMaker.EESumThreshold = cms.double(0.1)
        process.towerMakerPF.EESumThreshold = cms.double(0.1)
        process.towerMakerWithHO.EESumThreshold = cms.double(0.1)
        process.towerMaker.EEThreshold = cms.double(0.035)
        process.towerMakerPF.EEThreshold = cms.double(0.035)
        process.towerMakerWithHO.EEThreshold = cms.double(0.035)
        
        # Change all processes to use EcalRecHitsEK instead of EcalRecHitsEE
        process.EcalHaloData.EERecHitLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.JPTeidTight.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.calomuons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.conversionTrackCandidates.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.earlyMuons.CaloExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.earlyMuons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.earlyMuons.JetExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.ecalDrivenGsfElectrons.endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidLoose.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidRobustHighEnergy.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidRobustLoose.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidRobustTight.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidTight.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.gedGsfElectrons.endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.gedPhotons.mipVariableSet.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.gedPhotons.isolationSumsCalculatorSet.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.gsfElectrons.endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.interestingEleIsoDetIdEE.recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.interestingGamIsoDetIdEE.recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonMETValueMapProducer.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muons1stStep.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muons1stStep.JetExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muons1stStep.CaloExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics.JetExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics.CaloExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics1Leg.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics1Leg.JetExtractorPSet.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics1Leg.JetExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics1Leg.CaloExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.particleFlowSuperClusterECAL.regressionConfig.ecalRecHitsEE = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.pfElectronInterestingEcalDetIdEE.recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.pfPhotonInterestingEcalDetIdEE.recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.pfPhotonTranslator.endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.pfPhotonTranslator.EGPhotons = cms.string("mustachePhotons")
        process.photons.mipVariableSet.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.photons.isolationSumsCalculatorSet.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.uncleanedOnlyConversionTrackCandidates.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.uncleanedOnlyGsfElectrons.endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        ## The following ones don't work out of the box, so until they're fixed let them use the wrong collection
        #process.multi5x5BasicClustersCleaned.endcapHitCollection = cms.string('EcalRecHitsEK')
        #process.multi5x5BasicClustersUncleaned.endcapHitCollection = cms.string('EcalRecHitsEK')
        #process.correctedMulti5x5SuperClustersWithPreshower.recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        #process.uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower.recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEK")

    if hasattr(process,'validation_step'):
        process.ecalEndcapClusterTaskExtras.EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.ecalEndcapRecoSummary.recHitCollection_EE = cms.InputTag("ecalRecHit","EcalRecHitsEK")

    if hasattr(process,'FEVTDEBUGHLTEventContent'):
        process.FEVTDEBUGHLTEventContent.outputCommands.append('keep *_particleFlowRecHitHBHE_*_*')
        process.FEVTDEBUGHLTEventContent.outputCommands.append('keep *_particleFlowClusterHBHE_*_*')

    return process

def cust_2023SHCalNoExtPix(process):
    process=cust_2023MuonNoExtPix(process)
    if hasattr(process,'L1simulation_step'):
        process.simEcalTriggerPrimitiveDigis.BarrelOnly = cms.bool(True)
    if hasattr(process,'digitisation_step'):
        process.mix.digitizers.ecal.accumulatorType = cms.string('EcalPhaseIIDigiProducer')
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits","EcalHitsEK") )
        process.mix.mixObjects.mixCH.subdets.append( "EcalHitsEK" )
        process.simEcalUnsuppressedDigis = cms.EDAlias(
            mix = cms.VPSet(
            cms.PSet(type = cms.string('EBDigiCollection')),
            cms.PSet(type = cms.string('EEDigiCollection')),
            cms.PSet(type = cms.string('EKDigiCollection')),
            cms.PSet(type = cms.string('ESDigiCollection'))
            )
            )
        
    if hasattr(process,'raw2digi_step'):
        process.ecalDigis.FEDs = cms.vint32(
            # EE-:
            #601, 602, 603, 604, 605,
            #606, 607, 608, 609,
            # EB-:
            610, 611, 612, 613, 614, 615,
            616, 617, 618, 619, 620, 621,
            622, 623, 624, 625, 626, 627,
            # EB+:
            628, 629, 630, 631, 632, 633,
            634, 635, 636, 637, 638, 639,
            640, 641, 642, 643, 644, 645,
            # EE+:
            #646, 647, 648, 649, 650,
            #651, 652, 653, 654
            )
        print "RAW2DIGI only for EB FEDs"

    if hasattr(process,'reconstruction_step'):
    	process.ecalRecHit.EEuncalibRecHitCollection = cms.InputTag("","")
        process.reducedEcalRecHitsSequence.remove(process.reducedEcalRecHitsES)
        #remove the old EE pfrechit producer
        del process.particleFlowRecHitECAL.producers[1]
        process.particleFlowClusterEBEKMerger = cms.EDProducer('PFClusterCollectionMerger',
                                                               inputs = cms.VInputTag(cms.InputTag('particleFlowClusterECALUncorrected'),
                                                                                      cms.InputTag('particleFlowClusterEKUncorrected')
                                                                                      )
                                                               )   
        process.pfClusteringECAL.remove(process.particleFlowClusterECAL)
#        process.pfClusteringECAL.remove(process.particleFlowClusterECALWithTimeSelected)
        process.pfClusteringECAL += process.pfClusteringEK 
        process.pfClusteringECAL += process.particleFlowClusterEBEKMerger
        process.pfClusteringECAL += process.particleFlowClusterECAL        
        process.particleFlowClusterECAL.inputECAL = cms.InputTag('particleFlowClusterEBEKMerger')
        process.pfClusteringECAL += process.particleFlowClusterECAL
        #process.particleFlowCluster += process.pfClusteringEK
        
        #clone photons to mustache photons so we can compare back to old reco
        process.mustachePhotonCore = process.photonCore.clone(scHybridBarrelProducer = cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALBarrel"),scIslandEndcapProducer = cms.InputTag("particleFlowSuperClusterECAL","particleFlowSuperClusterECALEndcapWithPreshower"))
        process.mustachePhotons = process.photons.clone(photonCoreProducer = cms.InputTag('mustachePhotonCore'), endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEK"))        
        process.photonSequence += process.mustachePhotonCore
        process.photonSequence += process.mustachePhotons
        #point particle flow at the right photon collection     
        process.particleFlowBlock.elementImporters[2].source = cms.InputTag('mustachePhotons')
        process.gedPhotons.endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        
        process.towerMaker.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEK"))
        process.towerMakerPF.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEK"))
        process.towerMakerWithHO.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEK"))
        process.towerMaker.EESumThreshold = cms.double(0.1)
        process.towerMakerPF.EESumThreshold = cms.double(0.1)
        process.towerMakerWithHO.EESumThreshold = cms.double(0.1)
        process.towerMaker.EEThreshold = cms.double(0.035)
        process.towerMakerPF.EEThreshold = cms.double(0.035)
        process.towerMakerWithHO.EEThreshold = cms.double(0.035)
        
        # Change all processes to use EcalRecHitsEK instead of EcalRecHitsEE
        process.EcalHaloData.EERecHitLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.JPTeidTight.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.calomuons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.conversionTrackCandidates.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.earlyMuons.CaloExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.earlyMuons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.earlyMuons.JetExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.ecalDrivenGsfElectrons.endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidLoose.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidRobustHighEnergy.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidRobustLoose.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidRobustTight.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.eidTight.reducedEndcapRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.gedGsfElectrons.endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.gedPhotons.mipVariableSet.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.gedPhotons.isolationSumsCalculatorSet.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.gsfElectrons.endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.interestingEleIsoDetIdEE.recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.interestingGamIsoDetIdEE.recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonMETValueMapProducer.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muons1stStep.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muons1stStep.JetExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muons1stStep.CaloExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics.JetExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics.CaloExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics1Leg.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics1Leg.JetExtractorPSet.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics1Leg.JetExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.muonsFromCosmics1Leg.CaloExtractorPSet.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.particleFlowSuperClusterECAL.regressionConfig.ecalRecHitsEE = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.pfElectronInterestingEcalDetIdEE.recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.pfPhotonInterestingEcalDetIdEE.recHitsLabel = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.pfPhotonTranslator.endcapEcalHits = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.pfPhotonTranslator.EGPhotons = cms.string("mustachePhotons")
        process.photons.mipVariableSet.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.photons.isolationSumsCalculatorSet.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.uncleanedOnlyConversionTrackCandidates.endcapEcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.uncleanedOnlyGsfElectrons.endcapRecHitCollectionTag = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        ## The following ones don't work out of the box, so until they're fixed let them use the wrong collection
        #process.multi5x5BasicClustersCleaned.endcapHitCollection = cms.string('EcalRecHitsEK')
        #process.multi5x5BasicClustersUncleaned.endcapHitCollection = cms.string('EcalRecHitsEK')
        #process.correctedMulti5x5SuperClustersWithPreshower.recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        #process.uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower.recHitProducer = cms.InputTag("ecalRecHit","EcalRecHitsEK")

    if hasattr(process,'validation_step'):
        process.ecalEndcapClusterTaskExtras.EcalRecHitCollection = cms.InputTag("ecalRecHit","EcalRecHitsEK")
        process.ecalEndcapRecoSummary.recHitCollection_EE = cms.InputTag("ecalRecHit","EcalRecHitsEK")

    return process

def cust_2023HGCal_common(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=customise_gem2023(process)
    process=customise_rpc(process)
    process=jetCustoms.customise_jets(process)
    if hasattr(process,'L1simulation_step'):
    	process.simEcalTriggerPrimitiveDigis.BarrelOnly = cms.bool(True)
    if hasattr(process,'digitisation_step'):
    	process.mix.digitizers.ecal.accumulatorType = cms.string('EcalPhaseIIDigiProducer')
        process.load('SimGeneral.MixingModule.hgcalDigitizer_cfi')
        process.mix.digitizers.hgceeDigitizer=process.hgceeDigitizer
        process.mix.digitizers.hgchebackDigitizer=process.hgchebackDigitizer
        process.mix.digitizers.hgchefrontDigitizer=process.hgchefrontDigitizer
        # Also need to tell the MixingModule to make the correct collections available from
        # the pileup, even if not creating CrossingFrames.
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgceeDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgchebackDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.input.append( cms.InputTag("g4SimHits",process.hgchefrontDigitizer.hitCollection.value()) )
        process.mix.mixObjects.mixCH.subdets.append( process.hgceeDigitizer.hitCollection.value() )
        process.mix.mixObjects.mixCH.subdets.append( process.hgchebackDigitizer.hitCollection.value() )
        process.mix.mixObjects.mixCH.subdets.append( process.hgchefrontDigitizer.hitCollection.value() )
    if hasattr(process,'raw2digi_step'):
        process.ecalDigis.FEDs = cms.vint32(
            # EE-:
            #601, 602, 603, 604, 605,
            #606, 607, 608, 609,
            # EB-:
            610, 611, 612, 613, 614, 615,
            616, 617, 618, 619, 620, 621,
            622, 623, 624, 625, 626, 627,
            # EB+:
            628, 629, 630, 631, 632, 633,
            634, 635, 636, 637, 638, 639,
            640, 641, 642, 643, 644, 645,
            # EE+:
            #646, 647, 648, 649, 650,
            #651, 652, 653, 654
            )
        print "RAW2DIGI only for EB FEDs"
    if hasattr(process,'reconstruction_step'):
        process.particleFlowCluster += process.particleFlowRecHitHGC
        process.particleFlowCluster += process.particleFlowClusterHGC
        if hasattr(process,'particleFlowSuperClusterECAL'):
            process.particleFlowSuperClusterHGCEE = process.particleFlowSuperClusterECAL.clone()
            process.particleFlowSuperClusterHGCEE.useHGCEmPreID = cms.bool(True)
            process.particleFlowSuperClusterHGCEE.PFClusters = cms.InputTag('particleFlowClusterHGCEE')
            process.particleFlowSuperClusterHGCEE.use_preshower = cms.bool(False)
            process.particleFlowSuperClusterHGCEE.PFSuperClusterCollectionEndcapWithPreshower = cms.string('')
            process.particleFlowCluster += process.particleFlowSuperClusterHGCEE
            if hasattr(process,'ecalDrivenElectronSeeds'):
                process.ecalDrivenElectronSeeds.endcapSuperClusters = cms.InputTag('particleFlowSuperClusterHGCEE')
                process.ecalDrivenElectronSeeds.SeedConfiguration.endcapHCALClusters = cms.InputTag('particleFlowClusterHGCHEF')
                process.ecalDrivenElectronSeeds.SeedConfiguration.hOverEMethodEndcap = cms.int32(3)
                process.ecalDrivenElectronSeeds.SeedConfiguration.maxHOverEEndcaps = cms.double(0.2) 
                process.ecalDrivenElectronSeeds.SeedConfiguration.z2MinB = cms.double(-0.15)
                process.ecalDrivenElectronSeeds.SeedConfiguration.z2MaxB = cms.double(0.15)
                if hasattr(process,'ecalDrivenGsfElectrons'):
                    process.ecalDrivenGsfElectrons.hOverEMethodEndcap = cms.int32(3)
                    process.ecalDrivenGsfElectrons.hcalEndcapClusters = cms.InputTag('particleFlowClusterHGCHEF')
                    if hasattr(process,'gsfElectrons'):
                        process.gsfElectrons.hOverEMethodEndcap = cms.int32(3)
                        process.gsfElectrons.hcalEndcapClusters = cms.InputTag('particleFlowClusterHGCHEF')
        if hasattr(process,'particleFlowBlock'):
            process.particleFlowBlock.elementImporters.append( cms.PSet( importerName = cms.string('HGCECALClusterImporter'),
                                                                         source = cms.InputTag('particleFlowClusterHGCEE') ) )
            process.particleFlowBlock.linkDefinitions.append( cms.PSet( linkerName = cms.string('TrackAndHGCEELinker'),
                                                                        linkType = cms.string('TRACK:HGC_ECAL'),
                                                                        useKDTree = cms.bool(True) ) )
            process.particleFlowBlock.linkDefinitions.append( cms.PSet( linkerName = cms.string('TrackAndHGCHEFLinker'),
                                                                        linkType = cms.string('TRACK:HGC_HCALF'),
                                                                        useKDTree = cms.bool(True) ) )
            process.particleFlowBlock.linkDefinitions.append( cms.PSet( linkerName = cms.string('TrackAndHGCHEBLinker'),
                                                                        linkType = cms.string('TRACK:HGC_HCALB'),
                                                                        useKDTree = cms.bool(True) ) )
            process.particleFlowBlock.linkDefinitions.append( cms.PSet( linkType = cms.string('SC:HGC_ECAL'),
                                                                        SuperClusterMatchByRef = cms.bool(True),
                                                                        useKDTree = cms.bool(False),
                                                                        linkerName = cms.string('SCAndECALLinker') ) ) 
    #mod event content
    process.load('RecoLocalCalo.Configuration.hgcalLocalReco_EventContent_cff')
    if hasattr(process,'FEVTDEBUGHLTEventContent'):
        process.FEVTDEBUGHLTEventContent.outputCommands.extend(process.hgcalLocalRecoFEVT.outputCommands)
        process.FEVTDEBUGHLTEventContent.outputCommands.append('keep *_particleFlowSuperClusterHGCEE_*_*')
    if hasattr(process,'RECOSIMEventContent'):
        process.RECOSIMEventContent.outputCommands.extend(process.hgcalLocalRecoFEVT.outputCommands)
        process.RECOSIMEventContent.outputCommands.append('keep *_particleFlowSuperClusterHGCEE_*_*')
    return process

def cust_2023HGCal(process):
    process = cust_2023HGCal_common(process)
    return process

def cust_2023HGCalMuon(process):
    process = cust_2023HGCal_common(process)
    process = customise_me0(process)
    return process

def cust_2023HGCalV6Muon(process):
    """
    Customisation function for the Extended2023HGCalV6Muon geometry. Currently does
    exactly the same as the cust_2023HGCalMuon function but this could change in the
    future.
    """
    process = cust_2023HGCal_common(process)
    process = customise_me0(process)
    return process

def cust_2023SHCalTime(process):
    process=cust_2023SHCal(process)
    process=cust_shashlikTime(process)
    process=cust_ecalTime(process)    
    return process

def cust_2023Pixel(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=customise_gem2023(process)
    process=customise_rpc(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_2023Muon(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=customise_gem2023(process)
    process=customise_rpc(process)
    process=customise_me0(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_2023MuonNoExtPix(process):
    process=customisePostLS1(process)
    process=customiseBE5D(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5D(process)
    process=customise_gem2023(process)
    process=customise_rpc(process)
    process=customise_me0(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_2023Muondev(process):
    process=customisePostLS1(process)
    process=customiseBE5DPixel10Ddev(process)
    process=customise_HcalPhase2(process)
    process=customise_ev_BE5DPixel10D(process)
    process=customise_gem2023(process)
    process=customise_rpc(process)
    process=customise_me0(process)
    process=jetCustoms.customise_jets(process)
    return process

def cust_2023TTI(process):
    process=customisePostLS1(process)
    process=customiseTTI(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase0(process)
    process=customise_ev_l1tracker(process)
    return process

def cust_2023TTI_forHLT(process):
    process=customisePostLS1(process)
    process=customiseTTI(process)
    process=customiseBE5DPixel10D(process)
    process=customise_HcalPhase0(process)
    process=l1EventContent_TTI_forHLT(process)
    return process


def noCrossing(process):
    process=customise_NoCrossing(process)
    return process

##### clone aging.py here 
def agePixel(process,lumi):
    process=aging.agePixel(process,lumi)
    return process

def ageHcal(process,lumi):
    process=aging.ageHcal(process,lumi)
    return process

def ageEcal(process,lumi):
    process=aging.ageEcal(process,lumi)
    return process

def customise_aging_100(process):
    process=aging.customise_aging_100(process)
    return process

def customise_aging_200(process):
    process=aging.customise_aging_200(process)
    return process

def customise_aging_300(process):
    process=aging.customise_aging_300(process)
    return process

def customise_aging_400(process):
    process=aging.customise_aging_400(process)
    return process

def customise_aging_500(process):
    process=aging.customise_aging_500(process)
    return process

def customise_aging_600(process):
    process=aging.customise_aging_600(process)
    return process

def customise_aging_700(process):
    process=aging.customise_aging_700(process)
    return process


def customise_aging_1000(process):
    process=aging.customise_aging_1000(process)
    return process

def customise_aging_3000(process):
    process=aging.customise_aging_3000(process)
    return process

def customise_aging_ecalonly_300(process):
    process=aging.customise_aging_ecalonly_300(process)
    return process

def customise_aging_ecalonly_1000(process):
    process=aging.customise_aging_ecalonly_1000(process)
    return process

def customise_aging_ecalonly_3000(process):
    process=aging.customise_aging_ecalonly_3000(process)
    return process

def customise_aging_newpixel_1000(process):
    process=aging.customise_aging_newpixel_1000(process)
    return process

def customise_aging_newpixel_3000(process):
    process=aging.customise_aging_newpixel_3000(process)
    return process

def ecal_complete_aging(process):
    process=aging.ecal_complete_aging(process)
    return process

def turn_off_HE_aging(process):
    process=aging.turn_off_HE_aging(process)
    return process

def turn_off_HF_aging(process):
    process=aging.turn_off_HF_aging(process)
    return process

def turn_off_Pixel_aging(process):
    process=aging.turn_off_Pixel_aging(process)
    return process

def turn_on_Pixel_aging_1000(process):
    process=aging.turn_on_Pixel_aging_1000(process)
    return process

def hf_complete_aging(process):
    process=aging.hf_complete_aging(process)
    return process
    
def ecal_complete_aging_300(process):
    process=aging.ecal_complete_aging_300(process)
    return process

def ecal_complete_aging_1000(process):
    process=aging.ecal_complete_aging_1000(process)
    return process

def ecal_complete_aging_3000(process):
    process=aging.ecal_complete_aging_3000(process)
    return process

def fastsimDefault(process):
    return fastCustomiseDefault(process)

def fastsimPhase2(process):
    return fastCustomisePhase2(process)

def bsStudyStep1(process):
    process.VtxSmeared.MaxZ = 11.0
    process.VtxSmeared.MinZ = -11.0
    return process

def bsStudyStep2(process):
    process.initialStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.02),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.7)
        )
    process.highPtTripletStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.02),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.7)
        )
    process.lowPtQuadStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.02),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.2)
        )
    process.lowPtTripletStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.015),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.35)
        )
    process.detachedQuadStepSeeds.RegionFactoryPSet.RegionPSet = cms.PSet(
        precise = cms.bool(True),
        originRadius = cms.double(0.5),
        originHalfLength = cms.double(11.0),#nSigmaZ = cms.double(4.0),
        beamSpot = cms.InputTag("offlineBeamSpot"),
        ptMin = cms.double(0.3)
        )
    return process

def customise_noPixelDataloss(process):
    return cNoPixDataloss(process)

