import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras
#import SLHCUpgradeSimulations.Configuration.customise_PFlow as customise_PFlow

#GEN-SIM so far...
def customise(process):
    print "!!!You are using the SUPPORTED Flat version of the Phase2 Tracker !!!"
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
    if hasattr(process,'validation_step'):
        process=customise_Validation(process,float(n))
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
    
    # insert new InnerTracker pixel clusterizer
    process.load("RecoLocalTracker.Phase2ITPixelClusterizer.Phase2ITPixelClusterizer_cfi")
    process.phase2ITPixelClusters.src = cms.InputTag('simSiPixelDigis', "Pixel")
    process.phase2ITPixelClusters.MissCalibrate = cms.untracked.bool(False)

    # keep new clusters
    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT','RECOSIM']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):
            getattr(process,b).outputCommands.append('keep *_siPhase2Clusters_*_*')
            getattr(process,b).outputCommands.append('keep *_phase2ITPixelClusters_*_*')

    #use with latest pixel geometry
    process.ClusterShapeHitFilterESProducer.PixelShapeFile = cms.string('RecoPixelVertexing/PixelLowPtUtilities/data/pixelShape_Phase1Tk.par')
    # Need this line to stop error about missing siPixelDigis.
    process.MeasurementTrackerEvent.inactivePixelDetectorLabels = cms.VInputTag()

    process.InitialStepPreSplitting.remove(process.siPixelClusters)

    process.reconstruction.remove(process.castorreco)
    process.reconstruction.remove(process.CastorTowerReco)
    process.reconstruction.remove(process.ak5CastorJets)
    process.reconstruction.remove(process.ak5CastorJetID)
    process.reconstruction.remove(process.ak7CastorJets)
    #process.reconstruction.remove(process.ak7BasicJets)
    process.reconstruction.remove(process.ak7CastorJetID)

    # Need these until pixel templates are used
    process.load("SLHCUpgradeSimulations.Geometry.recoFromSimDigis_cff")
    process.siPixelClusters.src = cms.InputTag('simSiPixelDigis', "Pixel")

    # As in the phase1 tracking reconstruction,
    # Remove the pre-cluster-splitting clustering step
    # To be enabled later together with or after the jet core step is enabled
    # This snippet must be after the loading of recoFromSimDigis_cff    
    process.pixeltrackerlocalreco = cms.Sequence(
        process.siPhase2Clusters +
        process.phase2ITPixelClusters +
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

    # STILL TO DO (when the ph2 PF will be included):
    # Particle flow needs to know that the eta range has increased, for
    # when linking tracks to HF clusters
#    process=customise_PFlow.customise_extendedTrackerBarrel( process )

    process.MeasurementTrackerEvent.Phase2TrackerCluster1DProducer = cms.string('siPhase2Clusters')
    process.MeasurementTrackerEvent.stripClusterProducer = cms.string('')
 
    return process

def customise_condOverRides(process):
    process.load('SLHCUpgradeSimulations.Geometry.fakeConditions_phase2TkFlat_cff')
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
