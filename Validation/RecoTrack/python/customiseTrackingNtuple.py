import FWCore.ParameterSet.Config as cms
import six

def _label(tag):
    if hasattr(tag, "getModuleLabel"):
        t = tag
    else:
        t = cms.InputTag(tag)
    return t.getModuleLabel()+t.getProductInstanceLabel()

def customiseTrackingNtupleTool(process, isRECO = True, mergeIters = False):
    process.load("Validation.RecoTrack.trackingNtuple_cff")
    process.TFileService = cms.Service("TFileService",
        fileName = cms.string('trackingNtuple.root')
    )

    if process.trackingNtuple.includeSeeds.value():
        if isRECO:
            if not hasattr(process, "reconstruction_step"):
                raise Exception("TrackingNtuple includeSeeds=True needs reconstruction which is missing")
        else: #assumes HLT with PF iter
            if not hasattr(process, "HLTIterativeTrackingIter02"):
                raise Exception("TrackingNtuple includeSeeds=True needs HLTIterativeTrackingIter02 which is missing")

    # Replace validation_step with ntuplePath
    if not hasattr(process, "validation_step"):
        raise Exception("TrackingNtuple customise assumes process.validation_step exists")


    # Should replay mixing for pileup simhits?
    usePileupSimHits = hasattr(process, "mix") and hasattr(process.mix, "input") and len(process.mix.input.fileNames) > 0
#    process.eda = cms.EDAnalyzer("EventContentAnalyzer")

    if not isRECO:
        if not hasattr(process,"hltMultiTrackValidation"):
            process.load("Validation.RecoTrack.HLTmultiTrackValidator_cff")
        process.trackingNtupleSequence = process.hltMultiTrackValidation.copy()
        import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi as SiStripRecHitConverter_cfi
        process.hltSiStripRecHits = SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
            ClusterProducer = "hltSiStripRawToClustersFacility",
            StripCPE = "hltESPStripCPEfromTrackAngle:hltESPStripCPEfromTrackAngle"
        )
        process.trackingNtupleSequence.insert(0,process.trackingParticlesIntime+process.simHitTPAssocProducer)
        process.trackingNtupleSequence.remove(process.hltTrackValidator)
        process.trackingNtupleSequence += process.hltSiStripRecHits + process.trackingNtuple

    #combine all *StepTracks (TODO: write one for HLT)
    if mergeIters and isRECO:
        process.mergedStepTracks = cms.EDProducer("TrackSimpleMerger",
            src = cms.VInputTag(m.replace("Seeds", "Tracks").replace("seedTracks", "") for m in process.trackingNtuple.seedTracks)
        )
        process.trackingNtupleSequence.insert(0,process.mergedStepTracks)
        process.trackingNtuple.tracks = "mergedStepTracks"

    ntuplePath = cms.Path(process.trackingNtupleSequence)

    if process.trackingNtuple.includeAllHits and process.trackingNtuple.includeTrackingParticles and usePileupSimHits:
        ntuplePath.insert(0, cms.SequencePlaceholder("mix"))

        process.load("Validation.RecoTrack.crossingFramePSimHitToPSimHits_cfi")
        instanceLabels = [_label(tag) for tag in process.simHitTPAssocProducer.simHitSrc]
        process.crossingFramePSimHitToPSimHits.src = ["mix:"+l for l in instanceLabels]
        process.simHitTPAssocProducer.simHitSrc = ["crossingFramePSimHitToPSimHits:"+l for l in instanceLabels]
        process.trackingNtupleSequence.insert(0, process.crossingFramePSimHitToPSimHits)

    # Bit of a hack but works
    modifier = cms.Modifier()
    modifier._setChosen()
    modifier.toReplaceWith(process.prevalidation_step, ntuplePath)
    modifier.toReplaceWith(process.validation_step, cms.EndPath())

    # remove the validation_stepN and prevalidatin_stepN of phase2 validation...    
    for p in [process.paths_(), process.endpaths_()]:    
        for pathName, path in six.iteritems(p):    
            if "prevalidation_step" in pathName:    
                if len(pathName.replace("prevalidation_step", "")) > 0:    
                    modifier.toReplaceWith(path, cms.Path())    
            elif "validation_step" in pathName:    
                if len(pathName.replace("validation_step", "")) > 0:    
                    modifier.toReplaceWith(path, cms.EndPath())

    # Remove all output modules
    for outputModule in six.itervalues(process.outputModules_()):
        for path in six.itervalues(process.paths_()):
            path.remove(outputModule)
        for path in six.itervalues(process.endpaths_()):
            path.remove(outputModule)
        

    return process

def customiseTrackingNtuple(process):
    customiseTrackingNtupleTool(process, isRECO = True)
    return process

def customiseTrackingNtupleMergeIters(process):
    customiseTrackingNtupleTool(process, isRECO = True, mergeIters = True)
    return process

def customiseTrackingNtupleHLT(process):
    import Validation.RecoTrack.TrackValidation_cff as _TrackValidation_cff
    _seedProducers = [
        "hltIter0PFLowPixelSeedsFromPixelTracks",
        "hltIter1PFLowPixelSeedsFromPixelTracks",
        "hltIter2PFlowPixelSeeds",
        "hltDoubletRecoveryPFlowPixelSeeds"
    ]
    _candidatesProducers = [ 
        "hltIter0PFlowCkfTrackCandidates",
        "hltIter1PFlowCkfTrackCandidates",
        "hltIter2PFlowCkfTrackCandidates",
        "hltDoubletRecoveryPFlowCkfTrackCandidates"
    ]
    (_seedSelectors, _tmpTask) = _TrackValidation_cff._addSeedToTrackProducers(_seedProducers, globals())
    _seedSelectorsTask = cms.Task()
    for modName in _seedSelectors:
        if not hasattr(process, modName):
            setattr(process,modName, globals()[modName].clone(beamSpot = "hltOnlineBeamSpot"))
        _seedSelectorsTask.add(getattr(process, modName))

    customiseTrackingNtupleTool(process, isRECO = False)

    process.trackingNtupleSequence.insert(0,cms.Sequence(_seedSelectorsTask))
    if process.hltSiStripRawToClustersFacility.onDemand.value():
        #make sure that all iter tracking is done before running the ntuple-related modules
        process.trackingNtupleSequence.insert(0,process.hltMergedTracks)

    process.trackingNtuple.tracks = "hltMergedTracks"
    process.trackingNtuple.seedTracks = _seedSelectors
    process.trackingNtuple.trackCandidates = _candidatesProducers
    process.trackingNtuple.clusterTPMap = "hltTPClusterProducer"
    process.trackingNtuple.trackAssociator = "hltTrackAssociatorByHits"
    process.trackingNtuple.beamSpot = "hltOnlineBeamSpot"
    process.trackingNtuple.pixelRecHits = "hltSiPixelRecHits"
    process.trackingNtuple.stripRphiRecHits = "hltSiStripRecHits:rphiRecHit"
    process.trackingNtuple.stripStereoRecHits = "hltSiStripRecHits:stereoRecHit"
    process.trackingNtuple.stripMatchedRecHits = "hltSiStripRecHits:matchedRecHit"
    process.trackingNtuple.vertices = "hltPixelVertices"
    process.trackingNtuple.TTRHBuilder = "hltESPTTRHBWithTrackAngle"
    process.trackingNtuple.parametersDefiner = "hltLhcParametersDefinerForTP"
    process.trackingNtuple.includeMVA = False

    return process

def extendedContent(process):
    process.trackingParticlesIntime.intimeOnly = False
    process.trackingNtuple.includeOOT = True
    process.trackingNtuple.keepEleSimHits = True

    process.trackingNtuple.saveSimHitsP3 = True
    process.trackingNtuple.addSeedCurvCov = True

    return process
