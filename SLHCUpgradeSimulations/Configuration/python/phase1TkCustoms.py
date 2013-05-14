import FWCore.ParameterSet.Config as cms

from RecoTracker.IterativeTracking.iterativeTk_cff import *
from RecoTracker.IterativeTracking.ElectronSeeds_cff import *
from SLHCUpgradeSimulations.Configuration.customise_mixing import customise_pixelMixing_PU

def customise(process):
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    if hasattr(process,'reconstruction'):
        if hasattr(process,'mix'): 
            n=0
            if hasattr(process.mix,'input'):
                n=process.mix.input.nbPileupEvents.averageNumber.value()
        else:
            print 'phase1TkCustoms requires a --pileup option to cmsDriver to run the reconstruction'
            print 'Please provide one!'
            sys.exit(1)
        if n>0:
            process=customise_RecoForPU(process,float(n))
        else:
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
    process.mix.digitizers.pixel.thePixelColEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.thePixelChipEfficiency_FPix3 = cms.double(0.999)
    process.mix.digitizers.pixel.AddPixelInefficiencyFromPython = cms.bool(True)

    process=customise_pixelMixing_PU(process)
    return process


def customise_Reco(process):
    #use with latest pixel geometry
    process.ClusterShapeHitFilterESProducer.PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par')
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


    process.pixelTracks.SeedMergerPSet = cms.PSet(
        layerListName = cms.string('PixelSeedMergerQuadruplets'),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
        )

    
    #done
    return process


# DQM steps change
def customise_DQM(process):
    # We cut down the number of iterative tracking steps
    process.dqmoffline_step.remove(process.TrackMonStep3)
    process.dqmoffline_step.remove(process.TrackMonStep4)
    process.dqmoffline_step.remove(process.TrackMonStep5)
    process.dqmoffline_step.remove(process.TrackMonStep6)
    #
    process.dqmoffline_step.remove(process.muonAnalyzer)
    process.dqmoffline_step.remove(process.jetMETAnalyzer)
    process.dqmoffline_step.remove(process.TrackMonStep9)
    process.dqmoffline_step.remove(process.TrackMonStep10)
#    process.dqmoffline_step.remove(process.PixelTrackingRecHitsValid)

    #put isUpgrade flag==true
    process.SiPixelRawDataErrorSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelDigiSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelClusterSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelRecHitSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelTrackResidualSource.isUpgrade = cms.untracked.bool(True)
    process.SiPixelHitEfficiencySource.isUpgrade = cms.untracked.bool(True)

    from DQM.TrackingMonitor.customizeTrackingMonitorSeedNumber import customise_trackMon_IterativeTracking_PHASE1
    process=customise_trackMon_IterativeTracking_PHASE1(process)
    
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

def add_detailed_pixel_dqm(process):
    #enable modOn
    process.SiPixelRawDataErrorSource.modOn = cms.untracked.bool(True)
    process.SiPixelDigiSource.modOn = cms.untracked.bool(True)
    process.SiPixelClusterSource.modOn = cms.untracked.bool(True)
    process.SiPixelRecHitSource.modOn = cms.untracked.bool(True)
    process.SiPixelTrackResidualSource.modOn = cms.untracked.bool(True)
    process.SiPixelHitEfficiencySource.modOn = cms.untracked.bool(True)

    return process


def remove_pixel_ineff(process):
    if hasattr(process,'mix'):
        process.mix.digitizers.pixel.AddPixelInefficiencyFromPython = cms.bool(False) 

    return process
    

def customise_RecoForPU(process,pileup):

    #this code supports either 70 or 140 pileup configurations - should fix as to support 0
    nPU=70
    if pileup>105: nPU=140
    
    #use with latest pixel geometry
    process.ClusterShapeHitFilterESProducer.PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par')
    # Need this line to stop error about missing siPixelDigis.
    process.MeasurementTracker.inactivePixelDetectorLabels = cms.VInputTag()

    # new layer list (3/4 pixel seeding) in InitialStep and pixelTracks
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

    # New tracking.  This is really ugly because it redefines globalreco and reconstruction.
    # It can be removed if change one line in Configuration/StandardSequences/python/Reconstruction_cff.py
    # from RecoTracker_cff.py to RecoTrackerPhase1PU140_cff.py

    # remove all the tracking first
    itIndex=process.globalreco.index(process.trackingGlobalReco)
    grIndex=process.reconstruction.index(process.globalreco)

    process.reconstruction.remove(process.globalreco)
    process.globalreco.remove(process.iterTracking)
    process.globalreco.remove(process.electronSeedsSeq)
    process.reconstruction_fromRECO.remove(process.trackingGlobalReco)
    del process.iterTracking
    del process.ckftracks
    del process.ckftracks_woBH
    del process.ckftracks_wodEdX
    del process.ckftracks_plus_pixelless
    del process.trackingGlobalReco
    del process.electronSeedsSeq
    del process.InitialStep
    del process.LowPtTripletStep
    del process.PixelPairStep
    del process.DetachedTripletStep
    del process.MixedTripletStep
    del process.PixelLessStep
    del process.TobTecStep
    del process.earlyGeneralTracks
    del process.ConvStep
    # add the correct tracking back in
    process.load("RecoTracker.Configuration.RecoTrackerPhase1PU"+str(nPU)+"_cff")

    process.globalreco.insert(itIndex,process.trackingGlobalReco)
    process.reconstruction.insert(grIndex,process.globalreco)
    #Note process.reconstruction_fromRECO is broken
    
    # End of new tracking configuration which can be removed if new Reconstruction is used.


    process.reconstruction.remove(process.castorreco)
    process.reconstruction.remove(process.CastorTowerReco)
    process.reconstruction.remove(process.ak7BasicJets)
    process.reconstruction.remove(process.ak7CastorJetID)

    #the quadruplet merger configuration     
    process.load("RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff")
    process.pixelseedmergerlayers.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.BPix.HitProducer = cms.string("siPixelRecHits" )
    process.pixelseedmergerlayers.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.pixelseedmergerlayers.FPix.HitProducer = cms.string("siPixelRecHits" )    
    
    # Need these until pixel templates are used
    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    # PixelCPEGeneric #
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
    process.PixelCPEGenericESProducer.DoCosmics = False
    # CPE for other steps
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
    # Turn of template use in tracking (iterative steps handled inside their configs)
    process.mergedDuplicateTracks.TTRHBuilder = 'WithTrackAngle'
    process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
    process.muonSeededSeedsInOut.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.muonSeededTracksInOut.TTRHBuilder=cms.string('WithTrackAngle')
    process.muonSeededTracksOutIn.TTRHBuilder=cms.string('WithTrackAngle')
    process.muons1stStep.TrackerKinkFinderParameters.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.regionalCosmicTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.cosmicsVetoTracksRaw.TTRHBuilder=cms.string('WithTrackAngle')
    # End of pixel template needed section
    
    # Make pixelTracks use quadruplets
    process.pixelTracks.SeedMergerPSet = cms.PSet(
        layerListName = cms.string('PixelSeedMergerQuadruplets'),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
        )

    return process
