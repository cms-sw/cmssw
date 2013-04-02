import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.iterativeTk_cff import *
from RecoTracker.IterativeTracking.ElectronSeeds_cff import *

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

def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.siPixelRawData)
    process.digi2raw_step.remove(process.castorRawData)
    return process

def customise_RawToDigi(process):
    process.raw2digi_step.remove(process.siPixelDigis)
    process.raw2digi_step.remove(process.castorDigis)
    return process

def customise_Digi(process):
    process.mix.digitizers.pixel.MissCalibrate = False
    process.mix.digitizers.pixel.LorentzAngle_DB = False
    process.mix.digitizers.pixel.killModules = False
    process.mix.digitizers.pixel.useDB = False
    process.mix.digitizers.pixel.DeadModules_DB = False
    process.mix.digitizers.pixel.NumPixelBarrel = cms.int32(4)
    process.mix.digitizers.pixel.NumPixelEndcap = cms.int32(3)
    process.mix.digitizers.pixel.ThresholdInElectrons_FPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix = cms.double(2000.0)
    process.mix.digitizers.pixel.ThresholdInElectrons_BPix_L1 = cms.double(2000.0)
    process.mix.digitizers.pixel.thePixelColEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_BPix4 = cms.double(0.999)
    process.mix.digitizers.pixel.AddPixelInefficiencyFromPython = cms.bool(False)

    return process

def customise_Reco(process):
    # Need this line to stop error about missing siPixelDigis.
    process.MeasurementTracker.inactivePixelDetectorLabels = cms.VInputTag()
    # Next line is only in for the moment for debugging
    #process.load('Configuration.StandardSequences.Reconstruction_cff')
    #
    process.load("RecoTracker.IterativeTracking.HighPtTripletStep_cff")

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
                                                        'FPix1_neg+FPix2_neg+FPix3_neg' )

    process.highPtTripletStepSeedLayers.layerList = cms.vstring('BPix1+BPix2+BPix3',
                                            'BPix2+BPix3+BPix4',
                                            'BPix1+BPix3+BPix4',
                                            'BPix1+BPix2+BPix4',
                                            'BPix2+BPix3+FPix1_pos', 
					    'BPix2+BPix3+FPix1_neg',
                                            'BPix1+BPix2+FPix1_pos', 
					    'BPix1+BPix2+FPix1_neg',
                                            'BPix1+BPix3+FPix1_pos', 
					    'BPix1+BPix3+FPix1_neg',
                                            'BPix2+FPix1_pos+FPix2_pos', 
					    'BPix2+FPix1_neg+FPix2_neg',
                                            'BPix1+FPix1_pos+FPix2_pos', 
					    'BPix1+FPix1_neg+FPix2_neg',
                                            'BPix1+BPix2+FPix2_pos', 
					    'BPix1+BPix2+FPix2_neg',
                                            'FPix1_pos+FPix2_pos+FPix3_pos', 
					    'FPix1_neg+FPix2_neg+FPix3_neg',
                                            'BPix1+FPix2_pos+FPix3_pos', 
					    'BPix1+FPix2_neg+FPix3_neg',
                                            'BPix1+FPix1_pos+FPix3_pos', 
					    'BPix1+FPix1_neg+FPix3_neg' )
						 
    process.lowPtTripletStepSeedLayers.layerList = cms.vstring( 'BPix1+BPix2+BPix3',
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
	    						'FPix1_neg+FPix2_neg+FPix3_neg' )

    ## need changes to mixedtriplets step to use for imcreasing high eta efficiency
    process.mixedTripletStepClusters.oldClusterRemovalInfo = cms.InputTag("pixelPairStepClusters")
    process.mixedTripletStepClusters.trajectories = cms.InputTag("pixelPairStepTracks")
    process.mixedTripletStepClusters.overrideTrkQuals = cms.InputTag('pixelPairStepSelector','pixelPairStep')
    process.mixedTripletStepSeedsA.RegionFactoryPSet.RegionPSet.originRadius = 0.02
    process.mixedTripletStepSeedsB.RegionFactoryPSet.RegionPSet.originRadius = 0.02
    
    ## new layer list for mixed triplet step
    process.mixedTripletStepSeedLayersA.layerList = cms.vstring('BPix1+BPix2+BPix3', 
                                                                'BPix2+BPix3+BPix4',
                                                                'BPix1+BPix2+FPix1_pos', 
								'BPix1+BPix2+FPix1_neg', 
                                                                'BPix1+FPix1_pos+FPix2_pos', 
								'BPix1+FPix1_neg+FPix2_neg', 
                                                                'FPix1_pos+FPix2_pos+FPix3_pos', 
								'FPix1_neg+FPix2_neg+FPix3_neg', 
                                                                'BPix2+FPix1_pos+FPix2_pos', 
								'BPix2+FPix1_neg+FPix2_neg', 
                                                                'FPix2_pos+FPix3_pos+TEC1_pos', 
								'FPix2_neg+FPix3_neg+TEC1_neg',
                                                                'FPix3_pos+TEC2_pos+TEC3_pos', 
								'FPix3_neg+TEC2_neg+TEC3_neg' )

    #mixedTripletStepSeedLayersB.layerList = cms.vstring('BPix3+BPix4+TIB1', 'BPix3+BPix4+TIB2')
    ## switch off SeedB the easy way
    process.mixedTripletStepSeedLayersB.layerList = cms.vstring('BPix1+BPix2+BPix3')

    ## increased the max track candidates
    process.mixedTripletStepTrackCandidates.maxNSeeds = cms.uint32(150000)
    process.pixelPairStepTrackCandidates.maxNSeeds    = cms.uint32(150000)

    ######### FOR initialStepSeeds SeedMergerPSet ---->  mergeTriplets must be true  
    global RecoTracker
    from RecoTracker.TkTrackingRegions.GlobalTrackingRegionFromBeamSpot_cfi import RegionPsetFomBeamSpotBlock
    process.initialStepSeeds = RecoTracker.TkSeedGenerator.GlobalSeedsFromTriplets_cff.globalSeedsFromTriplets.clone(
      RegionFactoryPSet = RegionPsetFomBeamSpotBlock.clone(
        ComponentName = cms.string('GlobalRegionProducerFromBeamSpot'),
        RegionPSet = RegionPsetFomBeamSpotBlock.RegionPSet.clone(
          ptMin = 0.6,
          originRadius = 0.02,
          nSigmaZ = 4.0
        )
      ),
      SeedMergerPSet = cms.PSet(
	layerListName = cms.string('PixelSeedMergerQuadruplets'),
	addRemainingTriplets = cms.bool(False),
	mergeTriplets = cms.bool(True),
	ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
      )
    )
    process.initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.SeedComparitorPSet.ComponentName = 'LowPtClusterShapeSeedComparitor'
    
    # quadruplets in step0
    #process.initialStepSeeds.SeedMergerPSet.mergeTriplets       = cms.bool(True)

    # disconnect merger for stepOne and step 2 to have triplets merged
    #process.highPtTripletStepSeeds.SeedMergerPSet.mergeTriplets = cms.bool(False)
    #process.lowPtTripletStepSeeds.SeedMergerPSet.mergeTriplets  = cms.bool(False)
    #process.pixelPairStepSeeds.SeedMergerPSet.mergeTriplets     = cms.bool(False)
    #process.mixedTripletStepSeedsA.SeedMergerPSet.mergeTriplets = cms.bool(False)
    #process.mixedTripletStepSeedsB.SeedMergerPSet.mergeTriplets = cms.bool(False)

    # to avoid 'too many clusters'
    process.initialStepSeeds.ClusterCheckPSet.doClusterCheck       = cms.bool(False)
    process.highPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck = cms.bool(False)
    process.lowPtTripletStepSeeds.ClusterCheckPSet.doClusterCheck  = cms.bool(False)
    process.pixelPairStepSeeds.ClusterCheckPSet.doClusterCheck     = cms.bool(False)
    process.mixedTripletStepSeedsA.ClusterCheckPSet.doClusterCheck = cms.bool(False)
    process.mixedTripletStepSeedsB.ClusterCheckPSet.doClusterCheck = cms.bool(False)
    
    # avoid 'number of triples exceed maximum'
    process.pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement            = cms.uint32(0)
    process.initialStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement       = cms.uint32(0)
    process.highPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
    process.lowPtTripletStepSeeds.OrderedHitsFactoryPSet.GeneratorPSet.maxElement  = cms.uint32(0)
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

    process.lowPtTripletStepSelector.trackSelectors[0].dz_par1 = cms.vdouble(0.605, 4.0) # 0.65
    process.lowPtTripletStepSelector.trackSelectors[0].dz_par2 = cms.vdouble(0.42, 4.0) # 0.45
    process.lowPtTripletStepSelector.trackSelectors[0].d0_par1 = cms.vdouble(0.51, 4.0) # 0.55
    process.lowPtTripletStepSelector.trackSelectors[0].d0_par2 = cms.vdouble(0.51, 4.0) # 0.55
    process.lowPtTripletStepSelector.trackSelectors[1].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
    process.lowPtTripletStepSelector.trackSelectors[1].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.lowPtTripletStepSelector.trackSelectors[1].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
    process.lowPtTripletStepSelector.trackSelectors[1].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.lowPtTripletStepSelector.trackSelectors[2].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
    process.lowPtTripletStepSelector.trackSelectors[2].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.lowPtTripletStepSelector.trackSelectors[2].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
    process.lowPtTripletStepSelector.trackSelectors[2].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4

    process.highPtTripletStepSelector.trackSelectors[0].dz_par1 = cms.vdouble(0.605, 4.0) # 0.65
    process.highPtTripletStepSelector.trackSelectors[0].dz_par2 = cms.vdouble(0.42, 4.0) # 0.45
    process.highPtTripletStepSelector.trackSelectors[0].d0_par1 = cms.vdouble(0.51, 4.0) # 0.55
    process.highPtTripletStepSelector.trackSelectors[0].d0_par2 = cms.vdouble(0.51, 4.0) # 0.55
    process.highPtTripletStepSelector.trackSelectors[1].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
    process.highPtTripletStepSelector.trackSelectors[1].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.highPtTripletStepSelector.trackSelectors[1].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
    process.highPtTripletStepSelector.trackSelectors[1].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.highPtTripletStepSelector.trackSelectors[2].dz_par1 = cms.vdouble(0.325, 4.0) # 0.35
    process.highPtTripletStepSelector.trackSelectors[2].dz_par2 = cms.vdouble(0.372, 4.0) # 0.4
    process.highPtTripletStepSelector.trackSelectors[2].d0_par1 = cms.vdouble(0.279, 4.0) # 0.3
    process.highPtTripletStepSelector.trackSelectors[2].d0_par2 = cms.vdouble(0.372, 4.0) # 0.4
   
    # This STEPS should be added later #
    ## remove tracking steps 2-5 to speed up the job
    process.iterTracking.remove(process.DetachedTripletStep)
    process.iterTracking.remove(process.PixelLessStep)
    process.iterTracking.remove(process.TobTecStep)

    #modify the track merger accordingly
    #process.generalTracks.TrackProducers.remove(cms.InputTag('detachedTripletStepTracks'))
    #process.generalTracks.TrackProducers.remove(cms.InputTag('pixelLessStepTracks'))
    #process.generalTracks.TrackProducers.remove(cms.InputTag('tobTecStepTracks'))

    #process.generalTracks.selectedTrackQuals.remove(cms.InputTag("detachedTripletStep"))
    #process.generalTracks.selectedTrackQuals.remove(cms.InputTag("pixelLessStepSelector","pixelLessStep"))
    #process.generalTracks.selectedTrackQuals.remove(cms.InputTag("tobTecStepSelector","tobTecStep"))

    # Corrections for IterativeTracking adding HigPtTripletStep for Phase 1 # Sequence and  Tags #

    # Cloning or Modifing Steps # PixelPairStep and others ...    
    import RecoTracker.IterativeTracking.LowPtTripletStep_cff
    process.lowPtTripletStepClusters = RecoTracker.IterativeTracking.LowPtTripletStep_cff.lowPtTripletStepClusters.clone(
       oldClusterRemovalInfo = cms.InputTag("highPtTripletStepClusters")
    )
    
    process.lowPtTripletStepClusters.trajectories     = cms.InputTag("highPtTripletStepTracks")
    process.lowPtTripletStepClusters.overrideTrkQuals = cms.InputTag('highPtTripletStepSelector','highPtTripletStep')
    process.lowPtTripletStepTracks.AlgorithmName      = cms.string('iter2')
    process.pixelPairStepTracks.AlgorithmName         = cms.string('iter3')
    ## REMOVED BEFORE ##process.detachedTripletStepTracks.AlgorithmName = cms.string('iter4')

    # MergeTrackCollections #    
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
    
    # Modifying iterTracking Sequence # Adding HighPtTripletStep to iterTracking Sequence #
    from RecoTracker.IterativeTracking.HighPtTripletStep_cff import HighPtTripletStep
    process.iterTracking = cms.Sequence(InitialStep*
                                         HighPtTripletStep*
					 LowPtTripletStep*
					 PixelPairStep*
					 MixedTripletStep*
		                         earlyGeneralTracks* # Adjust
					 preDuplicateMergingGeneralTracks* # Adjust
					 #generalTracks* # Adjust
					 generalTracksSequence*
					 ConvStep*
					 conversionStepTracks )  
    process.preDuplicateMergingGeneralTracks.TrackProducers = cms.VInputTag(cms.InputTag("earlyGeneralTracks"))
    process.preDuplicateMergingGeneralTracks.selectedTrackQuals = cms.VInputTag(cms.InputTag("muonSeededTracksOutInSelector","muonSeededTracksOutInHighPurity"))
    process.preDuplicateMergingGeneralTracks.setsToMerge = cms.VPSet(cms.PSet(
        pQual = cms.bool(False),
        tLists = cms.vint32(0)
    ))
    process.preDuplicateMergingGeneralTracks.hasSelector = cms.vint32(0)
    process.mergedDuplicateTracks.TTRHBuilder  = 'WithTrackAngle'
    # PixelCPEGeneric #
    process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    #process.PixelCPEGenericESProducer.SmallPitch = False
    process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
    process.PixelCPEGenericESProducer.DoCosmics = False

    # CPE for other steps
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')

    #and clean up the conversions (which probably need work)
    process.convClusters.oldClusterRemovalInfo=cms.InputTag("mixedTripletStepClusters")
    process.convClusters.trajectories=cms.InputTag("mixedTripletStepTracks")
    process.convClusters.overrideTrkQuals= cms.InputTag("mixedTripletStep")
    
    # Corrections for Electron Seeds # Sequence and Mask # Tags #

    process.tripletElectronSeedLayers.BPix.skipClusters=cms.InputTag('mixedTripletStepSeedClusterMask')
    process.tripletElectronSeedLayers.FPix.skipClusters=cms.InputTag('mixedTripletStepSeedClusterMask')
    process.tripletElectronClusterMask.oldClusterRemovalInfo=cms.InputTag('mixedTripletStepSeedClusterMask')
    
    process.initialStepSeedClusterMask.oldClusterRemovalInfo=cms.InputTag("mixedTripletStepClusters") #step before pixelLess

    # removing pixelLessStep for now # Taking it out from newCombinedSeeds below #
    #process.newCombinedSeeds.seedCollections.remove( cms.InputTag('pixelLessStepSeeds'))
    # removing pixelLessStep for now # Taking it out from electronSeeds Sequence below #
    #process.electronSeedsSeq.remove(process.pixelLessStepSeedClusterMask)

    from RecoLocalTracker.SubCollectionProducers.SeedClusterRemover_cfi import seedClusterRemover
    process.highPtTripletStepSeedClusterMask = seedClusterRemover.clone(
    	trajectories = cms.InputTag("highPtTripletStepSeeds"),
    	oldClusterRemovalInfo = cms.InputTag("initialStepSeedClusterMask")
    )

    # Now highPtTripletStepSeedClusterMask will be before pixelPairStepSeedClusterMask #
    process.pixelPairStepSeedClusterMask.oldClusterRemovalInfo = cms.InputTag("highPtTripletStepSeedClusterMask")

    ### Not the Tracking uses the 2 seed collections separately. The merged seed collection is produced 
    ### for backward compatibility with electron reconstruction
    process.newCombinedSeeds.seedCollections = cms.VInputTag(cms.InputTag('initialStepSeeds'),
 							     cms.InputTag("highPtTripletStepSeeds"),
 							     cms.InputTag('pixelPairStepSeeds'),
 							     cms.InputTag('mixedTripletStepSeeds'),
 							     cms.InputTag('tripletElectronSeeds'),
 							     cms.InputTag('pixelPairElectronSeeds'),
 							     cms.InputTag('stripPairElectronSeeds')  )

    process.electronSeedsSeq = cms.Sequence( initialStepSeedClusterMask*
    				     	     process.highPtTripletStepSeedClusterMask*
    				     	     pixelPairStepSeedClusterMask*
    				     	     mixedTripletStepSeedClusterMask*
    				     	     tripletElectronSeeds*
    				     	     tripletElectronClusterMask*
    				     	     pixelPairElectronSeeds*
    				     	     stripPairElectronSeeds*
    				     	     newCombinedSeeds  )

    process.reconstruction.remove(process.castorreco)
    process.reconstruction.remove(process.CastorTowerReco)
    process.reconstruction.remove(process.ak7BasicJets)
    process.reconstruction.remove(process.ak7CastorJetID)

    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)

    #the quadruplet merger configuration     
    process.load("RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff")
    process.pixelseedmergerlayers.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.BPix.HitProducer = cms.string("siPixelRecHits" )
    process.pixelseedmergerlayers.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.FPix.HitProducer = cms.string("siPixelRecHits" )    
    
    process.highPtTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle') 

    process.initialStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.pixelPairStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.lowPtTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.convStepTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.mixedTripletStepTracks.TTRHBuilder=cms.string('WithTrackAngle')

    process.muons1stStep.TrackerKinkFinderParameters.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.regionalCosmicTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.cosmicsVetoTracksRaw.TTRHBuilder=cms.string('WithTrackAngle')

    #well, this needs to move input the default configs
    #SeedMergerPSet = cms.PSet(
    #    layerListName = cms.string('PixelSeedMergerQuadruplets'),
    #    addRemainingTriplets = cms.bool(False),
    #    mergeTriplets = cms.bool(False),
    #    ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
    #    )

    #process.regionalCosmicTrackerSeeds.SeedMergerPSet=SeedMergerPSet
    
    #done
    return process


# DQM steps change
def customise_DQM(process):
    # We cut down the number of iterative tracking steps
    process.dqmoffline_step.remove(process.TrackMonStep3)
    process.dqmoffline_step.remove(process.TrackMonStep5)
    process.dqmoffline_step.remove(process.TrackMonStep6)
    #
    process.dqmoffline_step.remove(process.muonAnalyzer)
    process.dqmoffline_step.remove(process.jetMETAnalyzer)
    process.dqmoffline_step.remove(process.TrackMonStep9)
    process.dqmoffline_step.remove(process.TrackMonStep10)
#    process.dqmoffline_step.remove(process.PixelTrackingRecHitsValid)
    return process

def customise_Validation(process):
    process.validation_step.remove(process.PixelTrackingRecHitsValid)
    # We don't run the HLT
    process.validation_step.remove(process.HLTSusyExoVal)
    process.validation_step.remove(process.hltHiggsValidator)
    process.validation_step.remove(process.relvalMuonBits)
    return process

def customise_harvesting(process):
    process.dqmHarvesting.remove(process.jetMETDQMOfflineClient)
    process.dqmHarvesting.remove(process.dataCertificationJetMET)
    process.dqmHarvesting.remove(process.sipixelEDAClient)
    process.dqmHarvesting.remove(process.sipixelCertification)
    return (process)        

def customise_condOverRides(process):
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_R30F12_cff')
    process.trackerTopologyConstants.pxb_layerStartBit = cms.uint32(18)
    process.trackerTopologyConstants.pxb_ladderMask = cms.uint32(1023)
    return process
