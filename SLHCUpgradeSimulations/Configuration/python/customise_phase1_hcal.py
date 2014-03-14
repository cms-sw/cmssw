import FWCore.ParameterSet.Config as cms

#common stuff here
def customise(process):
    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    process.load("SLHCUpgradeSimulations.Geometry.upgradeTracking_phase1_cff")

    process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    process.PixelCPEGenericESProducer.SmallPitch = False
    process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
    process.PixelCPEGenericESProducer.DoCosmics = False

    ## CPE for other steps
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
    process.initialStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.lowPtTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.pixelPairStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.detachedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.mixedTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.pixelLessStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.tobTecStepTracks.TTRHBuilder = cms.string('WithTrackAngle')
    process.highPtTripletStepTracks.TTRHBuilder = cms.string('WithTrackAngle')

    # Need these lines to stop some errors about missing siStripDigis collections.
    # should add them to fakeConditions_Phase1_cff
    process.MeasurementTracker.inactiveStripDetectorLabels = cms.VInputTag()
    process.MeasurementTracker.UseStripModuleQualityDB     = cms.bool(False)
    process.MeasurementTracker.UseStripAPVFiberQualityDB   = cms.bool(False)
    process.MeasurementTracker.UseStripStripQualityDB      = cms.bool(False)
    process.MeasurementTracker.UsePixelModuleQualityDB     = cms.bool(False)
    process.MeasurementTracker.UsePixelROCQualityDB        = cms.bool(False)

    process.muons.TrackerKinkFinderParameters.TrackerRecHitBuilder = cms.string('WithTrackAngle')
    # The SeedMergerPSet should be added to the following file for Phase 1
    # RecoTracker/SpecialSeedGenerators/python/CombinatorialSeedGeneratorForCosmicsRegionalReconstruction_cfi.py
    # but pixel layers are not used here for cosmic TODO: need Maria and Jan to do appropriate thing here
    from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import PixelSeedMergerQuadruplets
    process.regionalCosmicTrackerSeeds.SeedMergerPSet = cms.PSet(
                mergeTriplets = cms.bool(False),
                        ttrhBuilderLabel = cms.string( "PixelTTRHBuilderWithoutAngle" ),
                        addRemainingTriplets = cms.bool(False),
                        layerList = PixelSeedMergerQuadruplets
                        )
    process.regionalCosmicTracks.TTRHBuilder = cms.string('WithTrackAngle')


    ### back to standard job commands ##################################################
    process.DigiToRaw.remove(process.castorRawData)

    process.DigiToRaw.remove(process.siPixelRawData)
    process.RawToDigi.remove(process.siPixelDigis)


    if hasattr(process,'dqmoffline_step'):
        process.dqmoffline_step.remove(process.SiPixelTrackResidualSource)
        process.dqmoffline_step.remove(process.jetMETAnalyzer)
        process.dqmoffline_step.remove(process.hltMonMuBits)
        process.dqmoffline_step.remove(process.vbtfAnalyzer)
        process.dqmoffline_step.remove(process.hltResults)
        process.dqmoffline_step.remove(process.egHLTOffDQMSource)
        process.dqmoffline_step.remove(process.globalAnalyzer)
        process.dqmoffline_step.remove(process.jetMETHLTOfflineSource)
        ##
        process.dqmoffline_step.remove(process.TrackerCollisionTrackMon)
    if hasattr(process,'validation_step'):
        process.validation_step.remove(process.hltHITval)
        process.validation_step.remove(process.HLTSusyExoVal)
        process.validation_step.remove(process.relvalMuonBits)
        process.validation_step.remove(process.hltMuonValidator)
        #this takes forever - seems like an infinite loop that I didnt debug yet
        process.validation_step.remove(process.pixelDigisValid)
    else:
    ## removing large memory usage module if we don't need it
        process.pdigi.remove(process.mergedtruth)

#  HCAL Upgrade Geometry

    process.load("RecoLocalCalo.HcalRecProducers.HcalUpgradeReconstructor_cff")
    process.load("RecoJets.Configuration.CaloTowersRec_cff")
    process.load("RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi")
    process.load("RecoLocalCalo.Configuration.RecoLocalCalo_cff")

    process.ecalGlobalUncalibRecHit.EBdigiCollection = cms.InputTag("simEcalDigis","ebDigis")
    process.ecalGlobalUncalibRecHit.EEdigiCollection = cms.InputTag("simEcalDigis","eeDigis")
    process.ecalRecHit.ebDetIdToBeRecovered = cms.InputTag("","")
    process.ecalRecHit.eeDetIdToBeRecovered = cms.InputTag("","")
    process.ecalRecHit.eeFEToBeRecovered = cms.InputTag("","")
    process.ecalRecHit.ebFEToBeRecovered = cms.InputTag("","")
    process.ecalRecHit.recoverEBFE = cms.bool(False)
    process.ecalRecHit.recoverEEFE = cms.bool(False)

    process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hbhe_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_ho_cfi")
    process.load("RecoLocalCalo.HcalRecProducers.HcalSimpleReconstructor_hf_cfi")

    process.hbheprereco.digiLabel = "simHcalUnsuppressedDigis"
    process.horeco.digiLabel = "simHcalUnsuppressedDigis"
    process.hfreco.digiLabel = "simHcalUnsuppressedDigis"
    process.hcalupgradereco.digiLabel = "simHcalUnsuppressedDigis"

### Known alterations for Reco #####################################################
    delattr(process,"hbhereco")
    process.hbhereco = process.hbheprereco.clone()

    process.metrecoPlusHCALNoise.remove(process.BeamHaloSummary)
    process.metrecoPlusHCALNoise.remove(process.GlobalHaloData)
    process.metrecoPlusHCALNoise.remove(process.HcalHaloData)

    process.metrecoPlusHCALNoise.remove(process.hcalnoise)
    process.jetGlobalReco = cms.Sequence(process.recoJets*process.recoTrackJets)
    process.jetHighLevelReco = cms.Sequence(process.recoJetAssociations*process.recoPFJets*process.recoJPTJets)

### Place to add in the reco steps one by one ######################################
    process.calolocalreco = cms.Sequence(process.ecalGlobalUncalibRecHit+
                                process.ecalDetIdToBeRecovered+
                                process.ecalRecHit+
                                process.ecalCompactTrigPrim+
                                process.ecalTPSkim+
                                process.ecalPreshowerRecHit+
                                #process.hbheprereco+
                                process.hbhereco+
                                process.horeco+process.hfreco+process.hcalupgradereco+process.towerMaker
                                #+process.zdcreco
                                )
    process.localreco  = cms.Sequence(process.trackerlocalreco+
                                process.muonlocalreco+
                                process.calolocalreco+
                                process.castorreco+
                                process.lumiProducer
                                )
    process.globalreco = cms.Sequence(process.offlineBeamSpot
                          *process.recopixelvertexing
                          *process.trackingGlobalReco
                          #*process.hcalGlobalRecoSequence 
                          *process.particleFlowCluster
                          *process.ecalClusters
                          *process.caloTowersRec
                          *process.vertexreco
                          *process.egammaGlobalReco
                          *process.pfTrackingGlobalReco
                          *process.jetGlobalReco
                          *process.muonrecoComplete
                          *process.muoncosmicreco
                          *process.CastorFullReco
                          )
    process.highlevelreco = cms.Sequence(process.egammaHighLevelRecoPrePF
                             *process.particleFlowReco
                             *process.egammaHighLevelRecoPostPF
                             *process.jetHighLevelReco
                             *process.tautagging
                             *process.metrecoPlusHCALNoise
                             *process.btagging
                             *process.recoPFMET
                             *process.PFTau
                             *process.regionalCosmicTracksSeq
                             *process.muoncosmichighlevelreco
                             *process.reducedRecHits
                             )
    process.reconstruction = cms.Sequence(  process.localreco       *
                                        process.globalreco      *
                                        process.highlevelreco   *
                                        process.logErrorHarvester
                                        )
    # Clean out some unused sequences
    process.hcalLocalRecoSequence=cms.Sequence()
    process.hcalGlobalRecoSequence=cms.Sequence()

    return(process)


def customise_pu15_25ns(process):

    process=customise(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_HCal_cff")

### set the number of pileup
    process.mix.input.nbPileupEvents = cms.PSet(
        averageNumber = cms.double(15.0)
        )
    return (process)

def customise_3bx(process):
    process.mix.maxBunch=1
    process.mix.minBunch=-1
    return process

def customise_pu15_25ns_3bx(process):
    process=customise_pu15_25ns(process)
    process=customise_3bx(process)
    return process

#pileup specific stuff here
def customise_pu50_25ns(process):

    process=customise(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_HCal_cff")

### set the number of pileup
    process.mix.input.nbPileupEvents = cms.PSet(
        averageNumber = cms.double(50.0)
        )
    
### if doing inefficiency at <PU>=50
    process.simSiPixelDigis.AddPixelInefficiency = 20
    ## also for strips TIB inefficiency if we want
    ## TIB1,2 inefficiency at 20%
    #process.simSiStripDigis.Inefficiency = 20
    ## TIB1,2 inefficiency at 50%
    #process.simSiStripDigis.Inefficiency = 30
    ## TIB1,2 inefficiency at 99% (i.e. dead)
    #process.simSiStripDigis.Inefficiency = 40
    
    return (process)

def customise_pu50_25ns_3bx(process):
    process=customise_pu50_25ns(process)
    process=customise_3bx(process)
    return process

def customise_wo_pairs(process):

    process=customise(process)

    process.generalTracks.TrackProducers = (cms.InputTag('initialStepTracks'),
                      cms.InputTag('highPtTripletStepTracks'),
                      cms.InputTag('lowPtTripletStepTracks'),
                      cms.InputTag('mixedTripletStepTracks'))
    process.generalTracks.hasSelector=cms.vint32(1,1,1,1)
    process.generalTracks.selectedTrackQuals = cms.VInputTag(cms.InputTag("initialStepSelector","initialStep"),
                                       cms.InputTag("highPtTripletStepSelector","highPtTripletStep"),
                                       cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
                                       cms.InputTag("mixedTripletStep")
                                       )
    process.generalTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3), pQual=cms.bool(True) )
                             )

    process.newCombinedSeeds.seedCollections = cms.VInputTag(
          cms.InputTag('initialStepSeeds'),
          cms.InputTag('highPtTripletStepSeeds'),
          cms.InputTag('lowPtTripletStepSeeds')
    )

    process.mixedTripletStepClusters.oldClusterRemovalInfo = cms.InputTag("lowPtTripletStepClusters")
    process.mixedTripletStepClusters.trajectories = cms.InputTag("lowPtTripletStepTracks")
    process.mixedTripletStepClusters.overrideTrkQuals = cms.InputTag('lowPtTripletStepSelector','lowPtTripletStep')

    process.iterTracking.remove(process.PixelPairStep)
    return (process)

def turnOffLegacyPhiCorrection(process):
    process.simHcalUnsuppressedDigis.HcalReLabel.RelabelRules.CorrectPhi = cms.untracked.bool(False)
    process.HcalHardcodeGeometryEP.HcalReLabel.RelabelRules.CorrectPhi = cms.untracked.bool(False)
    process.HcalReLabel.RelabelRules.CorrectPhi = cms.untracked.bool(False)
    return (process)

def customise_pu15_25ns_wo_pairs(process):

    process=customise_wo_pairs(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_HCal_cff")

### set the number of pileup
    process.mix.input.nbPileupEvents = cms.PSet(
        averageNumber = cms.double(15.0)
        )
    return (process)
    

#pileup specific stuff here
def customise_pu50_25ns_wo_pairs(process):

    process=customise_wo_pairs(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_HCal_cff")

### set the number of pileup
    process.mix.input.nbPileupEvents = cms.PSet(
        averageNumber = cms.double(50.0)
        )


### if doing inefficiency at <PU>=50
    process.simSiPixelDigis.AddPixelInefficiency = 20
    ## also for strips TIB inefficiency if we want
    ## TIB1,2 inefficiency at 20%
    #process.simSiStripDigis.Inefficiency = 20
    ## TIB1,2 inefficiency at 50%
    #process.simSiStripDigis.Inefficiency = 30
    ## TIB1,2 inefficiency at 99% (i.e. dead)
    #process.simSiStripDigis.Inefficiency = 40

    return (process)

def saveAllProds(process):
    keys=process.outputModules.keys()
    for k in keys:
        mod=getattr(process,k)
        mod.outputCommands.extend(['keep *_simHcalUnsuppressedDigis_*_*',
                                   'keep *_hcalupgradereco_*_*',
                                   'keep *_hbhereco_*_*',
                                   'keep recoPFRecHits_particleFlowRecHitPS_*_*',
                                   'keep recoPFRecHits_particleFlowRecHitECAL_*_*',
                                   'keep recoPFBlocks_particleFlowBlock_*_*',
                                   'keep recoPFRecTracks_pfTrack_*_*',
                                   'keep recoPFClusters_particleFlowClusterPS_*_*',
                                   'keep recoPFClusters_particleFlowClusterECAL_*_*',
                                   'keep recoPFDisplacedVertexCandidates_particleFlowDisplacedVertexCandidate_*_*',
                                   'keep recoPFCandidates_particleFlow_*_*',
                                   'keep recoPFClusters_particleFlowClusterHCAL_*_*',
                                   'keep recoPFClusters_particleFlowClusterHCALUpgrade_*_*',
                                   'keep recoPFClusters_particleFlowClusterHO_*_*',
                                   'keep CaloTowersSorted_towerMakerPF_*_*',
                                   'keep recoPFCandidates_pfNoPileUp_*_*',
                                   'keep recoPFCandidates_pfAllPhotons_*_*',
                                   'keep recoPFRecHits_particleFlowRecHitHCAL_*_*',
                                   'keep recoPFRecHits_particleFlowRecHitHCALUpgrade_*_*',
                                   'keep recoPFRecHits_particleFlowRecHitHO_*_*',
                                   'keep recoPileUpPFCandidates_pfPileUp_*_*',
                                   'keep recoGsfPFRecTracks_pfTrackElec_*_*',
                                   'keep recoGsfPFRecTracks_pfTrackElec_Secondary_*',
                                   'keep recoPFCandidates_pfAllNeutralHadrons_*_*',
                                   'keep recoPFCandidates_pfAllChargedHadrons_*_*',
                                   'keep recoPFCandidates_particleFlow_electrons_*',
                                   'keep recoPFRecTracks_pfV0_*_*',
                                   'keep recoJetedmRefToBaseProdrecoTracksrecoTrackrecoTracksTorecoTrackedmrefhelperFindUsingAdvanceedmRefVectorsAssociationVector_ak5PFJetTracksAssociatorAtVertex_*_*',
                                   'keep recoPFRecTracks_pfConversions_*_*',
                                   'keep recoPFCandidateElectronExtras_particleFlow_*_*',
                                   'keep recoPFRecHits_particleFlowRecHitECAL_Cleaned_*',
                                   'keep recoPFV0s_pfV0_*_*',
                                   'keep recoPFCandidates_pfSelectedElectrons_*_*',
                                   'keep recoPFDisplacedVertexs_particleFlowDisplacedVertex_*_*',
                                   'keep recoPFRecTracks_pfDisplacedTrackerVertex_*_*',
                                   'keep recoPFConversions_pfConversions_*_*',
                                   'keep recoPFCandidates_particleFlow_CleanedPunchThroughNeutralHadrons_*',
                                   'keep recoPFDisplacedTrackerVertexs_pfDisplacedTrackerVertex_*_*',
                                   'keep recoPFCandidates_particleFlow_CleanedTrackerAndGlobalMuons_*',
                                   'keep recoPFCandidates_particleFlow_CleanedPunchThroughMuons_*',
                                   'keep recoPFCandidates_particleFlow_AddedMuonsAndHadrons_*',
                                   'keep recoPFMETs_pfMet_*_*',
                                   'keep recoPFCandidates_particleFlow_CleanedCosmicsMuons_*',
                                   'keep recoPFRecHits_particleFlowClusterHFHAD_Cleaned_*',
                                   'keep recoPFCandidates_particleFlow_CleanedFakeMuons_*',
                                   'keep recoPFRecHits_particleFlowClusterECAL_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowClusterHFEM_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowClusterHCAL_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowRecHitHCAL_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowClusterHCALUpgrade_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowRecHitHCALUpgrade_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowClusterHO_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowRecHitHO_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowClusterPS_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowRecHitHCAL_HFHAD_*',
                                   'keep recoPFRecHits_particleFlowRecHitPS_Cleaned_*',
                                   'keep recoPFRecHits_particleFlowRecHitHCAL_HFEM_*',
                                   'keep recoPFClusters_particleFlowClusterHFHAD_*_*',
                                   'keep recoPFClusters_particleFlowClusterHFEM_*_*',
                                   'keep recoPFCandidates_particleFlow_CleanedHF_*',
                                   'keep recoPFCandidates_pfSelectedPhotons_*_*'])
    return process
