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
    process.regionalCosmicTrackerSeeds.SeedMergerPSet = cms.PSet(
                mergeTriplets = cms.bool(False),
                        ttrhBuilderLabel = cms.string( "PixelTTRHBuilderWithoutAngle" ),
                        addRemainingTriplets = cms.bool(False),
                        layerListName = cms.string( "PixelSeedMergerQuadruplets" )
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


    return(process)


def customise_pu15_25ns(process):

    process=customise(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_cff")

### set the number of pileup
    process.mix.input.nbPileupEvents = cms.PSet(
        averageNumber = cms.double(15.0)
        )
    return (process)


#pileup specific stuff here
def customise_pu50_25ns(process):

    process=customise(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_cff")

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

def customise_pu15_25ns_wo_pairs(process):

    process=customise_wo_pairs(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_cff")

### set the number of pileup
    process.mix.input.nbPileupEvents = cms.PSet(
        averageNumber = cms.double(15.0)
        )
    return (process)
    

#pileup specific stuff here
def customise_pu50_25ns_wo_pairs(process):

    process=customise_wo_pairs(process)

    process.load("SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_cff")

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

