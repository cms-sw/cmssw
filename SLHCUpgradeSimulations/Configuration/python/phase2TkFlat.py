import FWCore.ParameterSet.Config as cms
#import SLHCUpgradeSimulations.Configuration.customise_PFlow as customise_PFlow

#GEN-SIM so far...
def customise(process):
    print "!!!You are using the SUPPORTED FLAT version of the Phase2 Tracker !!!"
    if hasattr(process,'DigiToRaw'):
        process=customise_DigiToRaw(process)
    if hasattr(process,'RawToDigi'):
        process=customise_RawToDigi(process)
    n=0
    if hasattr(process,'reconstruction') or hasattr(process,'dqmoffline_step'):
        if hasattr(process,'mix'):
            if hasattr(process.mix,'input'):
                n=process.mix.input.nbPileupEvents.averageNumber.value()
        else:
            print 'phase1TkCustoms requires a --pileup option to cmsDriver to run the reconstruction/dqm'
            print 'Please provide one!'
            sys.exit(1)
    if hasattr(process,'reconstruction'):
        process=customise_Reco(process,float(n))
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    process=customise_condOverRides(process)

    return process

def customise_Digi(process):
    process.digitisation_step.remove(process.mix.digitizers.pixel)
    process.load('SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi')
    process.mix.digitizers.pixel=process.phase2TrackerDigitizer
    process.mix.digitizers.strip.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof",
                         'g4SimHitsTrackerHitsPixelEndcapLowTof')
    #Check if mergedtruth is in the sequence first, could be taken out depending on cmsDriver options
    if hasattr(process.mix.digitizers,"mergedtruth") :
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIBHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTOBHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTECHighTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDLowTof"))
        process.mix.digitizers.mergedtruth.simHitCollections.tracker.remove( cms.InputTag("g4SimHits","TrackerHitsTIDHighTof"))

    # keep new digis
    alist=['FEVTDEBUG','FEVTDEBUGHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep Phase2TrackerDigiedmDetSetVector_*_*_*')
    return process


def customise_DigiToRaw(process):
    process.digi2raw_step.remove(process.siPixelRawData)
    process.digi2raw_step.remove(process.rpcpacker)
    return process

def customise_RawToDigi(process):
    process.raw2digi_step.remove(process.siPixelDigis)
    return process

def customise_Reco(process,pileup):
    # insert the new clusterizer
    process.load('SimTracker.SiPhase2Digitizer.phase2TrackerClusterizer_cfi')
    
    #process.load('RecoLocalTracker.SubCollectionProducers.jetCoreClusterSplitter_cfi')	
    #clustersTmp = 'siPixelClustersPreSplitting'
     # 0. Produce tmp clusters in the first place.
    #process.siPixelClustersPreSplitting = process.siPixelClusters.clone()
    #process.siPixelRecHitsPreSplitting = process.siPixelRecHits.clone()
    #process.siPixelRecHitsPreSplitting.src = clustersTmp
    #process.pixeltrackerlocalreco.replace(process.siPixelClusters, process.siPixelClustersPreSplitting)
    #process.pixeltrackerlocalreco.replace(process.siPixelRecHits, process.siPixelRecHitsPreSplitting)
    #process.clusterSummaryProducer.pixelClusters = clustersTmp
    itIndex = process.pixeltrackerlocalreco.index(process.siPixelClustersPreSplitting)
    process.pixeltrackerlocalreco.insert(itIndex, process.siPhase2Clusters)
    process.pixeltrackerlocalreco.remove(process.siPixelClustersPreSplitting)
    process.pixeltrackerlocalreco.remove(process.siPixelRecHitsPreSplitting)
    process.trackerlocalreco.remove(process.clusterSummaryProducer)
    # keep new clusters
    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_siPhase2Clusters_*_*')


    #use with latest pixel geometry
    process.ClusterShapeHitFilterESProducer.PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par')
    # Need this line to stop error about missing siPixelDigis.
    process.MeasurementTracker.inactivePixelDetectorLabels = cms.VInputTag()

    # new layer list (3/4 pixel seeding) in InitialStep and pixelTracks
    process.PixelLayerTriplets.layerList = cms.vstring('BPix1+BPix2+BPix3', 'BPix2+BPix3+BPix4',
						       'BPix2+BPix3+FPix1_pos', 'BPix2+BPix3+FPix1_neg',
						       'BPix1+BPix2+FPix1_pos', 'BPix1+BPix2+FPix1_neg',
						       'BPix2+FPix1_pos+FPix2_pos', 'BPix2+FPix1_neg+FPix2_neg',
						       'BPix1+FPix1_pos+FPix2_pos', 'BPix1+FPix1_neg+FPix2_neg',
						       'BPix1+FPix2_pos+FPix3_pos', 'BPix1+FPix2_neg+FPix3_neg',
						       'FPix1_pos+FPix2_pos+FPix3_pos', 'FPix1_neg+FPix2_neg+FPix3_neg',
						       'FPix2_pos+FPix3_pos+FPix4_pos', 'FPix2_neg+FPix3_neg+FPix4_neg',
						       'FPix3_pos+FPix4_pos+FPix5_pos', 'FPix3_neg+FPix4_neg+FPix5_neg',
						       'FPix4_pos+FPix5_pos+FPix6_pos', 'FPix4_neg+FPix5_neg+FPix6_neg',
						       'FPix5_pos+FPix6_pos+FPix7_pos', 'FPix5_neg+FPix6_neg+FPix7_neg',
						       'FPix6_pos+FPix7_pos+FPix8_pos', 'FPix6_neg+FPix7_neg+FPix8_neg',
						       'FPix6_pos+FPix7_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix9_neg')

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
    del process.earlyMuons
    del process.muonSeededStepCore
    del process.muonSeededStepExtra 
    del process.muonSeededStep
    del process.muonSeededStepDebug
    
    # add the correct tracking back in
    process.load("RecoTracker.Configuration.RecoTrackerPhase2BEPixel10D_cff")

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
    process.pixelseedmergerlayers.layerList = cms.vstring('BPix1+BPix2+BPix3+BPix4',
						       'BPix1+BPix2+BPix3+FPix1_pos','BPix1+BPix2+BPix3+FPix1_neg',
						       'BPix1+BPix2+FPix1_pos+FPix2_pos', 'BPix1+BPix2+FPix1_neg+FPix2_neg',
						       'BPix1+FPix1_pos+FPix2_pos+FPix3_pos', 'BPix1+FPix1_neg+FPix2_neg+FPix3_neg',
						       'FPix1_pos+FPix2_pos+FPix3_pos+FPix4_pos', 'FPix1_neg+FPix2_neg+FPix3_neg+FPix4_neg',
						       'FPix2_pos+FPix3_pos+FPix4_pos+FPix5_pos', 'FPix2_neg+FPix3_neg+FPix4_neg+FPix5_neg',
						       'FPix3_pos+FPix4_pos+FPix5_pos+FPix6_pos', 'FPix3_neg+FPix4_neg+FPix5_neg+FPix6_pos',
						       'FPix4_pos+FPix5_pos+FPix6_pos+FPix7_pos', 'FPix4_neg+FPix5_neg+FPix6_neg+FPix7_neg',
						       'FPix5_pos+FPix6_pos+FPix7_pos+FPix8_pos', 'FPix5_neg+FPix6_neg+FPix7_neg+FPix8_neg',
						       'FPix5_pos+FPix6_pos+FPix7_pos+FPix9_pos', 'FPix5_neg+FPix6_neg+FPix7_neg+FPix9_neg',
						       'FPix6_pos+FPix7_pos+FPix8_pos+FPix9_pos', 'FPix6_neg+FPix7_neg+FPix8_neg+FPix9_neg')
    
    
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
    process.muons1stStep.TrackerKinkFinderParameters.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.regionalCosmicTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.cosmicsVetoTracksRaw.TTRHBuilder=cms.string('WithTrackAngle')
    # End of pixel template needed section
    
    process.regionalCosmicTrackerSeeds.OrderedHitsFactoryPSet.LayerPSet.layerList  = cms.vstring('BPix9+BPix8')  # Optimize later
    process.regionalCosmicTrackerSeeds.OrderedHitsFactoryPSet.LayerPSet.BPix = cms.PSet(
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.006),
        useErrorsFromParam = cms.bool(True),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        skipClusters = cms.InputTag("pixelPairStepClusters"),
        hitErrorRPhi = cms.double(0.0027)
    )
    # Make pixelTracks use quadruplets
    process.pixelTracks.SeedMergerPSet = cms.PSet(
        layerListName = cms.string('PixelSeedMergerQuadruplets'),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
        )
    process.pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
    process.pixelTracks.FilterPSet.chi2 = cms.double(50.0)
    process.pixelTracks.FilterPSet.tipMax = cms.double(0.05)
    process.pixelTracks.RegionFactoryPSet.RegionPSet.originRadius =  cms.double(0.02)

    # Particle flow needs to know that the eta range has increased, for
    # when linking tracks to HF clusters
    process=customise_PFlow.customise_extendedTrackerBarrel( process )

 
    return process

def customise_condOverRides(process):
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkFlat_cff')
    return process


