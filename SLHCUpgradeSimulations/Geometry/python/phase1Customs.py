import FWCore.ParameterSet.Config as cms

def phase1Mods(process):

    
    # new layer list (3/4 pixel seeding) in stepZero
    process.pixellayertriplets.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
                                                        'BPix2+BPix3+BPix4',
                                                        'BPix1+BPix3+BPix4',
                                                        'BPix1+BPix2+BPix4',
                                                        'BPix2+BPix3+FPix1_pos',
                                                        'BPix2+BPix3+FPix1_neg',
                                                        'BPix1+BPix2+FPix1_pos',
                                                        'BPix1+BPix2+FPix1_neg',
                                                        'BPix2+FPix1_pos+FPix2_pos',
                                                        'BPix2+FPix1_neg+FPix2_neg',
                                                        'BPix1+FPix1_pos+FPix2_pos',
                                                        'BPix1+FPix1_neg+FPix2_neg',
                                                        'FPix1_pos+FPix2_pos+FPix3_pos',
                                                        'FPix1_neg+FPix2_neg+FPix3_neg'
                                                        )
    
## need changes to mixedtriplets step to use for imcreasing high eta efficiency
    
    process.mixedTripletStepClusters.oldClusterRemovalInfo = cms.InputTag("pixelPairStepClusters")
    process.mixedTripletStepClusters.trajectories = cms.InputTag("pixelPairStepTracks")
    process.mixedTripletStepClusters.overrideTrkQuals = cms.InputTag('pixelPairStepSelector','pixelPairStep')
    process.mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originRadius = 0.02
    process.mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.originRadius = 0.02
    
## new layer list for mixed triplet step
    process.mixedTripletStepSeedLayersA.layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
                                                                'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg', 
                                                                'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg', 
                                                                'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg', 
                                                                'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg', 
                                                                'FPix2_pos+FPix3_pos+TEC1_pos', 'FPix2_neg+FPix3_neg+TEC1_neg',
                                                                'FPix3_pos+TEC2_pos+TEC3_pos', 'FPix3_neg+TEC2_neg+TEC3_neg')
    
    #mixedTripletStepSeedLayersB.layerList = cms.vstring('BPix3+BPix4+TIB1', 'BPix3+BPix4+TIB2')
## switch off SeedB the easy way
    process.mixedTripletStepSeedLayersB.layerList = cms.vstring('BPix1+BPix2+BPix3')
## increased the max track candidates
    process.mixedTripletStepTrackCandidates.maxNSeeds = cms.uint32(150000)
    process.pixelPairStepTrackCandidates.maxNSeeds = cms.uint32(150000)
    
    # quadruplets in step0
    process.initialStepSeeds.SeedMergerPSet.mergeTriplets=cms.bool(True)

    # to avoid 'too many clusters'
    process.initialStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
    process.pixelPairStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
    process.mixedTripletStepSeedsA.ClusterCheckPSet.doClusterCheck = cms.bool(False)
    process.mixedTripletStepSeedsB.ClusterCheckPSet.doClusterCheck = cms.bool(False)
    
    # avoid 'number of triples exceed maximum'
    process.pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
    process.initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
    process.mixedTripletStepSeedsA.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
    process.mixedTripletStepSeedsB.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
    # avoid 'number of pairs exceed maximum'
    process.pixelPairStepSeeds.OrderedHitsFactoryPSet.maxElement =  cms.uint32(0)
    
    process.initialStepSelector.trackSelectors[0].dz_par1 = cms.vdouble(0.605, 4.0) # 0.65
    process.initialStepSelector.trackSelectors[0].dz_par2 = cms.vdouble(0.42, 4.0) # 0.45
    process.initialStepSelector.trackSelectors[0].d0_par1 = cms.vdouble(0.51, 4.0) # 0.55
    process.initialStepSelector.trackSelectors[0].d0_par2 = cms.vdouble(0.51, 4.0) # 0.55
    process.initialStepSelector.trackSelectors[1].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
    process.initialStepSelector.trackSelectors[1].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.initialStepSelector.trackSelectors[1].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
    process.initialStepSelector.trackSelectors[1].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.initialStepSelector.trackSelectors[2].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
    process.initialStepSelector.trackSelectors[2].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.initialStepSelector.trackSelectors[2].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
    process.initialStepSelector.trackSelectors[2].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
    
   
    ## remove tracking steps 2-5 to speed up the job
    process.iterTracking.remove(process.DetachedTripletStep)
    process.iterTracking.remove(process.PixelLessStep)
    process.iterTracking.remove(process.TobTecStep)

    #modify the track merger accordingly
    process.generalTracks.TrackProducers.remove(cms.InputTag('detachedTripletStepTracks'))
    process.generalTracks.TrackProducers.remove(cms.InputTag('pixelLessStepTracks'))
    process.generalTracks.TrackProducers.remove(cms.InputTag('tobTecStepTracks'))

    process.generalTracks.selectedTrackQuals.remove(cms.InputTag("detachedTripletStep"))
    process.generalTracks.selectedTrackQuals.remove(cms.InputTag("pixelLessStepSelector","pixelLessStep"))
    process.generalTracks.selectedTrackQuals.remove(cms.InputTag("tobTecStepSelector","tobTecStep"))

    process.generalTracks.setsToMerge = cms.VPSet( cms.PSet( tLists=cms.vint32(0,1,2,3), pQual=cms.bool(True) ))
    process.generalTracks.hasSelector=cms.vint32(1,1,1,1)

    #and clean up the conversions (which probably need work)
    process.convClusters.oldClusterRemovalInfo=cms.InputTag("mixedTripletStepClusters")
    process.convClusters.trajectories=cms.InputTag("mixedTripletStepTracks")
    process.convClusters.overrideTrkQuals= cms.InputTag("mixedTripletStep")


    
    process.initialStepSeedClusterMask.oldClusterRemovalInfo=cms.InputTag("mixedTripletStepClusters") #step before pixelLess

    process.tripletElectronSeedLayers.BPix.skipClusters=cms.InputTag('mixedTripletStepSeedClusterMask')
    process.tripletElectronSeedLayers.FPix.skipClusters=cms.InputTag('mixedTripletStepSeedClusterMask')
    process.tripletElectronClusterMask.oldClusterRemovalInfo=cms.InputTag('mixedTripletStepSeedClusterMask')

    process.newCombinedSeeds.seedCollections.remove( cms.InputTag('pixelLessStepSeeds'))

    process.electronSeedsSeq.remove(process.pixelLessStepSeedClusterMask)

    process.DigiToRaw.remove(process.siPixelRawData)
    process.DigiToRaw.remove(process.castorRawData)
    process.RawToDigi.remove(process.siPixelDigis)
    process.RawToDigi.remove(process.castorDigis)
    process.reconstruction.remove(process.castorreco)
    process.reconstruction.remove(process.CastorTowerReco)
    process.reconstruction.remove(process.ak7BasicJets)
    process.reconstruction.remove(process.ak7CastorJetID)
    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    process.load("RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff")
    process.pixelseedmergerlayers.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.BPix.HitProducer = cms.string("siPixelRecHits" )
    process.pixelseedmergerlayers.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.FPix.HitProducer = cms.string("siPixelRecHits" )
    
    process.pixelseedmergerlayers.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.BPix.HitProducer = cms.string("siPixelRecHits" )
    process.pixelseedmergerlayers.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.FPix.HitProducer = cms.string("siPixelRecHits" )
    
    
    process.initialStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.pixelPairStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.lowPtTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.convStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.mixedTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    
    process.muons1stStep.TrackerKinkFinderParameters.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.regionalCosmicTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.cosmicsVetoTracksRaw.TTRHBuilder=cms.string('WithTrackAngle')

    #well, this needs to move input the default configs
    SeedMergerPSet = cms.PSet(
        layerListName = cms.string('PixelSeedMergerQuadruplets'),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(False),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
        )
    
    process.regionalCosmicTrackerSeeds.SeedMergerPSet=SeedMergerPSet
    process.pixelPairStepSeeds.SeedMergerPSet.mergeTriplets = cms.bool(False)

    
    #done
    return process
