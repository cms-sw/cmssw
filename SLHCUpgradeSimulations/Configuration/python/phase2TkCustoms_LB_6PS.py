import FWCore.ParameterSet.Config as cms
#GEN-SIM so far...
def customise(process):
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process)
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'dqmoffline_step'):
        process=customise_DQM(process)
    if hasattr(process,'dqmHarvesting'):
        process=customise_harvesting(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process)
    process=customise_condOverRides(process)
    
    return process

def customise_Digi(process):
    process.mix.digitizers.pixel.MissCalibrate = False
    process.mix.digitizers.pixel.LorentzAngle_DB = False
    process.mix.digitizers.pixel.killModules = False
    process.mix.digitizers.pixel.useDB = False
    process.mix.digitizers.pixel.DeadModules_DB = False
    process.mix.digitizers.pixel.NumPixelBarrel = cms.int32(12)
    process.mix.digitizers.pixel.NumPixelEndcap = cms.int32(3)
    process.mix.digitizers.pixel.ThresholdInElectrons_FPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix_L1 = cms.double(2000.0)
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelColEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.AddPixelInefficiencyFromPython = cms.bool(False)
    process.mix.digitizers.strip.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof",
                         'g4SimHitsTrackerHitsPixelEndcapLowTof')
    process.digitisation_step.remove(process.simSiStripDigiSimLink)
    process.mergedtruth.simHitCollections.tracker = []
    return process


def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.siPixelRawData)
    process.digi2raw_step.remove(process.rpcpacker)
    return process

def customise_RawToDigi(process):
    process.raw2digi_step.remove(process.siPixelDigis)
    return process

def customise_Reco(process):
    ## need changes to mixedtriplets step to use for imcreasing high eta efficiency
    process.reconstruction.remove(process.pixelLessStepSeedClusterMask)
    process.reconstruction.remove(process.castorreco)
    process.reconstruction.remove(process.CastorTowerReco)
    process.reconstruction.remove(process.ak7BasicJets)
    process.reconstruction.remove(process.ak7CastorJetID)
    #process.iterTracking.remove(process.PixelLessStep)
    #process.iterTracking.remove(process.TobTecStep)
    process.MixedTripletStep.remove(process.mixedTripletStepSeedsB)
    process.mixedTripletStepSeeds = cms.EDProducer("SeedCombiner",
        seedCollections = cms.VInputTag(cms.InputTag("mixedTripletStepSeedsA"))
    )
    process.load("RecoTracker.IterativeTracking.HighPtTripletStep_cff")
    from RecoTracker.IterativeTracking.HighPtTripletStep_cff import HighPtTripletStep
    process.iterTracking = cms.Sequence(process.InitialStep*
			    process.HighPtTripletStep*
                            process.LowPtTripletStep*
                            process.PixelPairStep*
                            process.DetachedTripletStep*
                            process.MixedTripletStep*
                            #process.PixelLessStep*
                            #process.TobTecStep*
                            process.earlyGeneralTracks*
                            process.muonSeededStep*
                            process.preDuplicateMergingGeneralTracks*
                            process.generalTracksSequence*
                            process.ConvStep*
                            process.conversionStepTracks
                            )




    process.convClusters.oldClusterRemovalInfo=cms.InputTag("mixedTripletStepClusters")
    process.convClusters.trajectories=cms.InputTag("mixedTripletStepTracks")
    process.convClusters.overrideTrkQuals= cms.InputTag("mixedTripletStep")
    process.mixedTripletStepSeedLayersA.layerList = cms.vstring('BPix1+BPix2+BPix3', 
        'BPix1+BPix2+FPix1_pos', 
        'BPix1+BPix2+FPix1_neg', 
        'BPix1+FPix1_pos+FPix2_pos', 
        'BPix1+FPix1_neg+FPix2_neg', 
        'BPix2+FPix1_pos+FPix2_pos', 
        'BPix2+FPix1_neg+FPix2_neg')

    process.earlyGeneralTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3,4), pQual=cms.bool(True) ))
    process.earlyGeneralTracks.hasSelector=cms.vint32(1,1,1,1,1)
    process.earlyGeneralTracks.selectedTrackQuals = cms.VInputTag(
         cms.InputTag("initialStepSelector","initialStep"),
         cms.InputTag("highPtTripletStepSelector","highPtTripletStep"),
         cms.InputTag("lowPtTripletStepSelector","lowPtTripletStep"),
         cms.InputTag("pixelPairStepSelector","pixelPairStep"),
         cms.InputTag("mixedTripletStep")
    )
    process.earlyGeneralTracks.TrackProducers = cms.VInputTag(
         cms.InputTag("initialStepTracks"),
         cms.InputTag("highPtTripletStepTracks"),
         cms.InputTag("lowPtTripletStepTracks"),
         cms.InputTag("pixelPairStepTracks"),
         cms.InputTag("mixedTripletStepTracks")
    )

    process.tripletElectronSeedLayers.BPix.skipClusters=cms.InputTag('mixedTripletStepSeedClusterMask')
    process.tripletElectronSeedLayers.FPix.skipClusters=cms.InputTag('mixedTripletStepSeedClusterMask')
    process.tripletElectronClusterMask.oldClusterRemovalInfo=cms.InputTag('mixedTripletStepSeedClusterMask')

    process.initialStepSeedClusterMask.oldClusterRemovalInfo=cms.InputTag("mixedTripletStepClusters") 
    process.newCombinedSeeds.seedCollections = cms.VInputTag(cms.InputTag('initialStepSeeds'),
                                                             cms.InputTag("highPtTripletStepSeeds"),
                                                             cms.InputTag('pixelPairStepSeeds'),
                                                             cms.InputTag('mixedTripletStepSeeds'),
                                                             cms.InputTag('tripletElectronSeeds'),
                                                             cms.InputTag('pixelPairElectronSeeds'), 
                                                             cms.InputTag('stripPairElectronSeeds')  )
    process.stripPairElectronSeedLayers.layerList = cms.vstring('BPix4+BPix5') # Optimize later
    process.stripPairElectronSeedLayers.BPix = cms.PSet(
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        skipClusters = cms.InputTag("pixelPairStepClusters"),
    )
    process.regionalCosmicTrackerSeeds.OrderedHitsFactoryPSet.LayerPSet.layerList  = cms.vstring('BPix12+BPix11')  # Optimize later
    process.regionalCosmicTrackerSeeds.OrderedHitsFactoryPSet.LayerPSet.BPix = cms.PSet(
        HitProducer = cms.string('siPixelRecHits'),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        skipClusters = cms.InputTag("pixelPairStepClusters"),
    )
    from RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff import PixelSeedMergerQuadruplets
    process.pixelTracks.SeedMergerPSet = cms.PSet(
        layerList = PixelSeedMergerQuadruplets,
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
        )
    process.initialStepSeedClusterMask.oldClusterRemovalInfo=cms.InputTag("mixedTripletStepClusters")
    
    # Need this line to stop error about missing siPixelDigis.
    process.MeasurementTracker.inactivePixelDetectorLabels = cms.VInputTag()
    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    # Use with latest pixel geometry. Only used for seeds, so we can use the Phase1Tk file.
    # We will need to turn it off for any steps that use the outer pixels as seeds.
    process.ClusterShapeHitFilterESProducer.PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par')
    # Now make sure we us CPE Generic
    process.mergedDuplicateTracks.TTRHBuilder  = 'WithTrackAngle'
    process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
    process.PixelCPEGenericESProducer.DoCosmics = False
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
    #the quadruplet merger configuration     
    # from this PSet the quadruplet merger uses only the layer list so these could probably be removed
    PixelSeedMergerQuadruplets.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    PixelSeedMergerQuadruplets.BPix.HitProducer = cms.string("siPixelRecHits" )
    PixelSeedMergerQuadruplets.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    PixelSeedMergerQuadruplets.FPix.HitProducer = cms.string("siPixelRecHits" )

    process.highPtTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.detachedTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.initialStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.pixelPairStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.lowPtTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.convStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.mixedTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.muonSeededSeedsInOut.TrackerRecHitBuilder = cms.string('WithTrackAngle')
    process.muonSeededTracksInOut.TTRHBuilder = cms.string('WithTrackAngle')
    process.muons1stStep.TrackerKinkFinderParameters.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.regionalCosmicTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.cosmicsVetoTracksRaw.TTRHBuilder=cms.string('WithTrackAngle')

    return process

def customise_condOverRides(process):
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_LongBarrel6PS_cff')
    process.trackerNumberingSLHCGeometry.layerNumberPXB = cms.uint32(20)
    process.trackerTopologyConstants.pxb_layerStartBit = cms.uint32(20)
    process.trackerTopologyConstants.pxb_ladderStartBit = cms.uint32(12)
    process.trackerTopologyConstants.pxb_moduleStartBit = cms.uint32(2)
    process.trackerTopologyConstants.pxb_layerMask = cms.uint32(15)
    process.trackerTopologyConstants.pxb_ladderMask = cms.uint32(255)
    process.trackerTopologyConstants.pxb_moduleMask = cms.uint32(1023)
    process.trackerTopologyConstants.pxf_diskStartBit = cms.uint32(18)
    process.trackerTopologyConstants.pxf_bladeStartBit = cms.uint32(12)
    process.trackerTopologyConstants.pxf_panelStartBit = cms.uint32(10)
    process.trackerTopologyConstants.pxf_moduleMask = cms.uint32(255)
    return process


def l1EventContent(process):
    #extend the event content

    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep PSimHits_g4SimHits_*_*')
            getattr(process,b).outputCommands.append('keep SimTracks_g4SimHits_*_*')
            getattr(process,b).outputCommands.append('keep SimVertexs_g4SimHits_*_*')
            getattr(process,b).outputCommands.append('keep *_simSiPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_genParticles_*_*')
            getattr(process,b).outputCommands.append('keep *_L1TkBeams_*_*')
            getattr(process,b).outputCommands.append('keep *_L1TkClustersFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_L1TkClustersFromSimHits_*_*')
            getattr(process,b).outputCommands.append('keep *_L1TkStubsFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_L1TkStubsFromSimHits_*_*')
            getattr(process,b).outputCommands.append('keep *_siPixelRecHits_*_*')
            #drop some bigger collections we don't think we need
            getattr(process,b).outputCommands.append('drop PSimHits_g4SimHits_EcalHitsEB_*')
            getattr(process,b).outputCommands.append('drop PSimHits_g4SimHits_EcalHitsEE_*')
            getattr(process,b).outputCommands.append('drop *_L1TkStubsFromSimHits_StubsFail_*')
    return process

