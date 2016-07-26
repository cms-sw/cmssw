import FWCore.ParameterSet.Config as cms
#import SLHCUpgradeSimulations.Configuration.customise_PFlow as customise_PFlow

#GEN-SIM so far...
def customise(process):
    print "!!!You are using the SUPPORTED Tilted version of the Phase2 Tracker !!!"
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
    if hasattr(process,'digitisationTkOnly_step'):
        process=customise_DigiTkOnly(process)
    if hasattr(process,'digitisation_step'):
        process=customise_Digi(process)
    if hasattr(process,'validation_step'):
        process=customise_Validation(process,float(n))
    process=customise_condOverRides(process)

    return process


def customise_DigiTkOnly(process):
    process.load('Configuration.StandardSequences.Digi_cff')
    process.doAllDigi = cms.Sequence()
    process.load('SimGeneral.MixingModule.mixObjects_cfi')
    process.digitisationTkOnly_step.remove(process.mix.mixObjects.mixCH)
    del process.simCastorDigis
    del process.simEcalUnsuppressedDigis
    del process.simHcalUnsuppressedDigis
    process.mix.digitizers = cms.PSet(process.theDigitizersValid)
    del process.mix.digitizers.ecal
    del process.mix.digitizers.hcal
    del process.mix.digitizers.castor
    process.digitisationTkOnly_step.remove(process.mix.digitizers.pixel)
    process.load('SimTracker.SiPhase2Digitizer.phase2TrackerDigitizer_cfi')
    process.mix.digitizers.pixel=process.phase2TrackerDigitizer
    process.mix.digitizers.strip.ROUList = cms.vstring("g4SimHitsTrackerHitsPixelBarrelLowTof",
                         'g4SimHitsTrackerHitsPixelEndcapLowTof')
    #Check if mergedtruth is in the sequence first, could be taken out depending on cmsDriver options
    if hasattr(process.mix.digitizers,"mergedtruth") :
        process.mix.digitizers.mergedtruth.simHitCollections.muon = cms.VInputTag( )
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
    
    # keep new clusters
    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_siPhase2Clusters_*_*')

    #use with latest pixel geometry
    process.ClusterShapeHitFilterESProducer.PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par')
    # Need this line to stop error about missing siPixelDigis.
    process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()

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
    itIndex=process.globalreco_tracking.index(process.trackingGlobalReco)
    grIndex=process.globalreco.index(process.globalreco_tracking)

    process.globalreco.remove(process.globalreco_tracking)
    process.globalreco_tracking.remove(process.iterTracking)
    process.globalreco_tracking.remove(process.electronSeedsSeq)
    process.reconstruction_fromRECO.remove(process.trackingGlobalReco)

    process.reconstruction_fromRECO.remove(process.initialStepSeedClusterMask)
    process.reconstruction_fromRECO.remove(process.initialStepSeedLayers)
    process.reconstruction_fromRECO.remove(process.initialStepSeeds)
    process.reconstruction_fromRECO.remove(process.initialStepTrackCandidates)
    process.reconstruction_fromRECO.remove(process.initialStepTracks)

    process.reconstruction_fromRECO.remove(process.lowPtTripletStepClusters)
    process.reconstruction_fromRECO.remove(process.lowPtTripletStepSeedLayers)
    process.reconstruction_fromRECO.remove(process.lowPtTripletStepSeeds)
    process.reconstruction_fromRECO.remove(process.lowPtTripletStep)
    process.reconstruction_fromRECO.remove(process.lowPtTripletStepTrackCandidates)
    process.reconstruction_fromRECO.remove(process.lowPtTripletStepTracks)

    process.reconstruction_fromRECO.remove(process.pixelPairStepSeedClusterMask)
    process.reconstruction_fromRECO.remove(process.pixelPairStepClusters)
    process.reconstruction_fromRECO.remove(process.pixelPairStepSeeds)
    process.reconstruction_fromRECO.remove(process.pixelPairStepSeedLayers)
    process.reconstruction_fromRECO.remove(process.pixelPairStep)
    process.reconstruction_fromRECO.remove(process.pixelPairStepTrackCandidates)
    process.reconstruction_fromRECO.remove(process.pixelPairStepTracks)

    process.reconstruction_fromRECO.remove(process.convClusters)
    process.reconstruction_fromRECO.remove(process.convLayerPairs)
    process.reconstruction_fromRECO.remove(process.convStepSelector)
    process.reconstruction_fromRECO.remove(process.convTrackCandidates)
    process.reconstruction_fromRECO.remove(process.convStepTracks)
    process.reconstruction_fromRECO.remove(process.photonConvTrajSeedFromSingleLeg)

    process.reconstruction_fromRECO.remove(process.muonSeededSeedsInOut)
    process.reconstruction_fromRECO.remove(process.muonSeededTrackCandidatesInOut)
    process.reconstruction_fromRECO.remove(process.muonSeededTracksInOut)

    process.reconstruction_fromRECO.remove(process.newCombinedSeeds)
    process.reconstruction_fromRECO.remove(process.preDuplicateMergingGeneralTracks)
    process.reconstruction_fromRECO.remove(process.tripletElectronClusterMask)
    process.reconstruction_fromRECO.remove(process.tripletElectronSeedLayers)
    process.reconstruction_fromRECO.remove(process.tripletElectronSeeds)

    process.reconstruction_fromRECO.remove(process.detachedQuadStep)
    process.reconstruction_fromRECO.remove(process.detachedQuadStepClusters)
    process.reconstruction_fromRECO.remove(process.detachedQuadStepSeedLayers)
    process.reconstruction_fromRECO.remove(process.detachedQuadStepSeeds)
    process.reconstruction_fromRECO.remove(process.detachedQuadStepTrackCandidates)
    process.reconstruction_fromRECO.remove(process.detachedQuadStepTracks)
    process.reconstruction_fromRECO.remove(process.detachedQuadStepSelector)

    process.reconstruction_fromRECO.remove(process.highPtTripletStepClusters)
    process.reconstruction_fromRECO.remove(process.highPtTripletStepSeedLayers)
    process.reconstruction_fromRECO.remove(process.highPtTripletStepSeeds)
    process.reconstruction_fromRECO.remove(process.highPtTripletStepTrackCandidates)
    process.reconstruction_fromRECO.remove(process.highPtTripletStepTracks)
    process.reconstruction_fromRECO.remove(process.highPtTripletStepSelector)

    process.reconstruction_fromRECO.remove(process.lowPtQuadStepClusters)
    process.reconstruction_fromRECO.remove(process.lowPtQuadStepSeedLayers)
    process.reconstruction_fromRECO.remove(process.lowPtQuadStepSeeds)
    process.reconstruction_fromRECO.remove(process.lowPtQuadStepTrackCandidates)
    process.reconstruction_fromRECO.remove(process.lowPtQuadStepTracks)

    del process.iterTracking
    del process.iterTrackingEarly
    del process.ckftracks
    del process.ckftracks_woBH
    del process.ckftracks_wodEdX
    del process.ckftracks_plus_pixelless
    del process.trackingGlobalReco
    del process.electronSeedsSeq
    del process.InitialStep
    del process.HighPtTripletStep
    del process.LowPtQuadStep
    del process.LowPtTripletStep
    del process.DetachedQuadStep
    del process.MixedTripletStep
    del process.PixelPairStep
    del process.TobTecStep
    del process.earlyGeneralTracks
    #del process.earlyMuons
    del process.muonSeededStep
    del process.muonSeededStepCore
    del process.muonSeededStepCoreInOut
    del process.muonSeededStepExtra 
    del process.muonSeededStepDebug
    del process.muonSeededStepDebugInOut
    del process.ConvStep
    
    # add the correct tracking back in
    process.load("RecoTracker.Configuration.RecoTrackerPhase2Tracker_cff")

    process.globalreco_tracking.insert(itIndex,process.trackingGlobalReco)
    process.globalreco.insert(grIndex,process.globalreco_tracking)
    #Note process.reconstruction_fromRECO is broken
    
    # End of new tracking configuration which can be removed if new Reconstruction is used.

    process.InitialStepPreSplitting.remove(process.siPixelClusters)

    process.reconstruction.remove(process.castorreco)
    process.reconstruction.remove(process.CastorTowerReco)
    process.reconstruction.remove(process.ak5CastorJets)
    process.reconstruction.remove(process.ak5CastorJetID)
    process.reconstruction.remove(process.ak7CastorJets)
    #process.reconstruction.remove(process.ak7BasicJets)
    process.reconstruction.remove(process.ak7CastorJetID)

    #the quadruplet merger configuration     
    process.load("RecoPixelVertexing.PixelTriplets.quadrupletseedmerging_cff")
    process.PixelSeedMergerQuadruplets.BPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.PixelSeedMergerQuadruplets.BPix.HitProducer = cms.string("siPixelRecHits" )
    process.PixelSeedMergerQuadruplets.FPix.TTRHBuilder = cms.string("PixelTTRHBuilderWithoutAngle" )
    process.PixelSeedMergerQuadruplets.FPix.HitProducer = cms.string("siPixelRecHits" )
    process.PixelSeedMergerQuadruplets.layerList = cms.vstring('BPix1+BPix2+BPix3+BPix4',
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
    process.siPixelClusters.src = cms.InputTag('simSiPixelDigis', "Pixel")

    # As in the phase1 tracking reconstruction,
    # Remove the pre-cluster-splitting clustering step
    # To be enabled later together with or after the jet core step is enabled
    # This snippet must be after the loading of recoFromSimDigis_cff    
    process.pixeltrackerlocalreco = cms.Sequence(
        process.siPhase2Clusters +
        process.siPixelClusters +
        process.siPixelRecHits
    )
    process.clusterSummaryProducer.pixelClusters = "siPixelClusters"
    process.globalreco_tracking.replace(process.MeasurementTrackerEventPreSplitting, process.MeasurementTrackerEvent)
    process.globalreco_tracking.replace(process.siPixelClusterShapeCachePreSplitting, process.siPixelClusterShapeCache)

    # As in the phase1 tracking reconstruction,
    # Enable, for now, pixel tracks and vertices
    # To be removed later together with the cluster splitting
    process.globalreco_tracking.replace(process.standalonemuontracking,
                                        process.standalonemuontracking+process.recopixelvertexing)
    process.initialStepSelector.vertices = "pixelVertices"
    process.highPtTripletStepSelector.vertices = "pixelVertices"
    process.lowPtQuadStepSelector.vertices = "pixelVertices"
    process.lowPtTripletStepSelector.vertices = "pixelVertices"
    process.detachedQuadStepSelector.vertices = "pixelVertices"
    process.mixedTripletStepSelector.vertices = "pixelVertices"
    process.pixelPairStepSeeds.RegionFactoryPSet.RegionPSet.VertexCollection = "pixelVertices"
    process.pixelPairStepSelector.vertices = "pixelVertices"
    process.tobTecStepSelector.vertices = "pixelVertices"
    process.muonSeededTracksInOutSelector.vertices = "pixelVertices"
    process.muonSeededTracksOutInSelector.vertices = "pixelVertices"
    process.duplicateTrackClassifier.vertices = "pixelVertices"
    process.convStepSelector.vertices = "pixelVertices"
    process.ak4CaloJetsForTrk.srcPVs = "pixelVertices"
    process.muonSeededTracksOutInDisplacedClassifier.vertices = "pixelVertices"
    process.duplicateDisplacedTrackClassifier.vertices = "pixelVertices"

    # PixelCPEGeneric #
    process.PixelCPEGenericESProducer.useLAWidthFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.Upgrade = cms.bool(True)
    process.PixelCPEGenericESProducer.UseErrorsFromTemplates = cms.bool(False)
    process.PixelCPEGenericESProducer.LoadTemplatesFromDB = cms.bool(False)
    process.PixelCPEGenericESProducer.TruncatePixelCharge = cms.bool(False)
    process.PixelCPEGenericESProducer.IrradiationBiasCorrection = False
    process.PixelCPEGenericESProducer.DoCosmics = False
    process.templates.DoLorentz = cms.bool(False)
    process.templates.LoadTemplatesFromDB = cms.bool(False)
    # CPE for other steps
    process.siPixelRecHits.CPE = cms.string('PixelCPEGeneric')
    # Turn of template use in tracking (iterative steps handled inside their configs)
    process.duplicateTrackCandidates.ttrhBuilderName = 'WithTrackAngle'
    process.mergedDuplicateTracks.TTRHBuilder = 'WithTrackAngle'
    process.ctfWithMaterialTracks.TTRHBuilder = 'WithTrackAngle'
    process.muonSeededSeedsInOut.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.muonSeededTracksInOut.TTRHBuilder=cms.string('WithTrackAngle')
    process.muons1stStep.TrackerKinkFinderParameters.TrackerRecHitBuilder=cms.string('WithTrackAngle')
    process.regionalCosmicTracks.TTRHBuilder=cms.string('WithTrackAngle')
    process.cosmicsVetoTracksRaw.TTRHBuilder=cms.string('WithTrackAngle')
    # End of pixel template needed section
    
    process.regionalCosmicTrackerSeedingLayers.layerList  = cms.vstring('BPix9+BPix8')  # Optimize later
    process.regionalCosmicTrackerSeedingLayers.BPix = cms.PSet(
        HitProducer = cms.string('siPixelRecHits'),
        hitErrorRZ = cms.double(0.006),
        useErrorsFromParam = cms.bool(True),
        TTRHBuilder = cms.string('TTRHBuilderWithoutAngle4PixelPairs'),
        skipClusters = cms.InputTag("pixelPairStepClusters"),
        hitErrorRPhi = cms.double(0.0027)
    )
    # Make pixelTracks use quadruplets
    process.pixelTracks.SeedMergerPSet = cms.PSet(
        layerList = cms.PSet(refToPSet_ = cms.string('PixelSeedMergerQuadruplets')),
        addRemainingTriplets = cms.bool(False),
        mergeTriplets = cms.bool(True),
        ttrhBuilderLabel = cms.string('PixelTTRHBuilderWithoutAngle')
        )
    process.pixelTracks.OrderedHitsFactoryPSet.GeneratorPSet.maxElement = cms.uint32(0)
    process.pixelTracks.FilterPSet.chi2 = cms.double(50.0)
    process.pixelTracks.FilterPSet.tipMax = cms.double(0.05)
    process.pixelTracks.RegionFactoryPSet.RegionPSet.originRadius =  cms.double(0.02)

    process.preDuplicateMergingDisplacedTracks.inputClassifiers.remove("muonSeededTracksInOutClassifier")
    process.preDuplicateMergingDisplacedTracks.trackProducers.remove("muonSeededTracksInOut")

    process.caloTowerForTrk.hbheInput = cms.InputTag("hbheUpgradeReco")
    process.caloTowerForTrk.hfInput = cms.InputTag("hfUpgradeReco")

    # STILL TO DO (when the ph2 PF will be included):
    # Particle flow needs to know that the eta range has increased, for
    # when linking tracks to HF clusters
#    process=customise_PFlow.customise_extendedTrackerBarrel( process )

    process.MeasurementTrackerEvent.Phase2TrackerCluster1DProducer = cms.string('siPhase2Clusters')
    process.MeasurementTrackerEvent.stripClusterProducer = cms.string('')
    # FIXME::process.electronSeedsSeq broken
    process.ckftracks.remove(process.electronSeedsSeq)
 
    return process

def customise_condOverRides(process):
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkTilted_cff')
    return process


def customise_Validation(process,pileup):

    process.pixelDigisValid.src = cms.InputTag('simSiPixelDigis', "Pixel")
    if hasattr(process,'tpClusterProducer'):
        process.tpClusterProducer.pixelSimLinkSrc = cms.InputTag("simSiPixelDigis", "Pixel")
        process.tpClusterProducer.phase2OTSimLinkSrc  = cms.InputTag("simSiPixelDigis","Tracker")

    if hasattr(process,'simHitTPAssocProducer'):
        process.simHitTPAssocProducer.simHitSrc=cms.VInputTag(cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
                                                              cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"))

    if hasattr(process,'trackingParticleNumberOfLayersProducer'):
        process.trackingParticleNumberOfLayersProducer.simHits=cms.VInputTag(cms.InputTag("g4SimHits","TrackerHitsPixelBarrelLowTof"),
                                                               cms.InputTag("g4SimHits","TrackerHitsPixelEndcapLowTof"))

    return process
