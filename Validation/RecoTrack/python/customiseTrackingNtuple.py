import FWCore.ParameterSet.Config as cms

def _label(tag):
    if hasattr(tag, "getModuleLabel"):
        t = tag
    else:
        t = cms.InputTag(tag)
    return t.getModuleLabel()+t.getProductInstanceLabel()

def _usePileupSimHits(process):
    return hasattr(process, "mix") and hasattr(process.mix, "input") and len(process.mix.input.fileNames) > 0

def customiseTrackingNtupleTool(process, isRECO = True, mergeIters = False):
    process.load("Validation.RecoTrack.trackingNtuple_cff")
    process.TFileService = cms.Service("TFileService",
        fileName = cms.string('trackingNtuple.root')
    )

    if process.trackingNtuple.includeSeeds.value():
        if isRECO:
            if not hasattr(process, "reconstruction_step"):
                raise Exception("TrackingNtuple includeSeeds=True needs reconstruction which is missing")
        else: #assumes HLT sequence
            if not (hasattr(process, "HLTIterativeTrackingIter02") or hasattr(process, "HLTTrackingSequence")):
                raise Exception("TrackingNtuple includeSeeds=True needs HLTIterativeTrackingIter02 or HLTTrackingSequence which is missing")

    # Replace validation_step with ntuplePath
    if not hasattr(process, "validation_step"):
        raise Exception("TrackingNtuple customise assumes process.validation_step exists")


    if not isRECO:
        if not hasattr(process,"hltMultiTrackValidation"):
            process.load("Validation.RecoTrack.HLTmultiTrackValidator_cff")
        process.trackingNtupleSequence = cms.Sequence(process.hltMultiTrackValidationTask)
        process.trackingNtupleSequence.insert(0,process.trackingParticlesIntime+process.simHitTPAssocProducer)

        if hasattr(process, "HLTIterativeTrackingIter02"):
            if not hasattr(process, "hltSiStripRecHits"):
                import RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitConverter_cfi as SiStripRecHitConverter_cfi
                process.hltSiStripRecHits = SiStripRecHitConverter_cfi.siStripMatchedRecHits.clone(
                    ClusterProducer = "hltSiStripRawToClustersFacility",
                    StripCPE = "hltESPStripCPEfromTrackAngle:hltESPStripCPEfromTrackAngle"
                )
            else:
                if not process.hltSiStripRecHits.doMatching.value():
                    process.hltSiStripRecHits.doMatching = True
            process.trackingNtupleSequence += process.hltSiStripRawToClustersFacility
            process.trackingNtupleSequence += process.hltSiStripRecHits
        elif hasattr(process, "HLTTrackingSequence"):
            if not hasattr(process, "hltSiPhase2RecHits"):
                import HLTrigger.Configuration.HLT_75e33.modules.hltSiPhase2RecHits_cfi as _mod
                process.hltSiPhase2RecHits = _mod.hltSiPhase2RecHits.clone()
            process.trackingNtupleSequence += process.hltSiPhase2Clusters
            process.trackingNtupleSequence += process.hltSiPhase2RecHits

        process.trackingNtupleSequence += process.trackingNtuple

    #combine all *StepTracks (TODO: write one for HLT)
    if mergeIters and isRECO:
        import RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi as _mod
        process.mergedStepTracks = _mod.TrackCollectionMerger.clone(
            trackProducers = cms.VInputTag(m.replace("Seeds", "Tracks").replace("seedTracks", "") for m in process.trackingNtuple.seedTracks),
            inputClassifiers = cms.vstring(m.replace("StepSeeds", "Step").replace("seedTracks", "").replace("dSeeds", "dTracks")
                                           .replace("InOut", "InOutClassifier").replace("tIn", "tInClassifier")
                                           for m in process.trackingNtuple.seedTracks),
            minQuality = "any",
            enableMerging = False
        )
        process.trackingNtupleSequence.insert(0,process.mergedStepTracks)
        process.trackingNtuple.tracks = "mergedStepTracks"
        process.trackingNtuple.includeMVA = True
        process.trackingNtuple.trackMVAs = ["mergedStepTracks"]

    ntuplePath = cms.Path(process.trackingNtupleSequence)

    if process.trackingNtuple.includeAllHits and process.trackingNtuple.includeTrackingParticles and _usePileupSimHits(process):
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
        for pathName, path in p.items():    
            if "prevalidation_step" in pathName:    
                if len(pathName.replace("prevalidation_step", "")) > 0:    
                    modifier.toReplaceWith(path, cms.Path())    
            elif "validation_step" in pathName:    
                if len(pathName.replace("validation_step", "")) > 0:    
                    modifier.toReplaceWith(path, cms.EndPath())

    # Remove all output modules
    for outputModule in process.outputModules_().values():
        for path in process.paths_().values():
            path.remove(outputModule)
        for path in process.endpaths_().values():
            path.remove(outputModule)
        

    return process

def customiseTrackingNtuple(process):
    customiseTrackingNtupleTool(process, isRECO = True)
    return process

def customiseTrackingNtupleMergeIters(process):
    customiseTrackingNtupleTool(process, isRECO = True, mergeIters = True)
    return process

from Validation.RecoTrack.plotting.ntupleEnum import Algo as _algo
def customiseTrackingNtupleHLT(process):
    import Validation.RecoTrack.TrackValidation_cff as _TrackValidation_cff
    _seedProducers = cms.PSet(
        names = cms.vstring("hltIter0PFLowPixelSeedsFromPixelTracks", "hltDoubletRecoveryPFlowPixelSeeds")
    )
    from Configuration.Eras.Modifier_trackingPhase2PU140_cff import trackingPhase2PU140
    trackingPhase2PU140.toModify(_seedProducers, names = ["hltInitialStepSeeds", "hltHighPtTripletStepSeeds"])
    # the following modifiers are only phase-2, trackingPhase2PU140 is not repeated
    from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
    from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
    from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
    (singleIterPatatrack & ~trackingLST).toModify(_seedProducers, names = ["hltInitialStepSeeds"])
    (singleIterPatatrack & trackingLST & ~seedingLST).toModify(_seedProducers, names = ["hltInputLST", "hltInitialStepTrackCandidates"])
    (singleIterPatatrack & trackingLST & seedingLST).toModify(_seedProducers, names = ["hltInitialStepTrajectorySeedsLST"])
    (~singleIterPatatrack & trackingLST & ~seedingLST).toModify(_seedProducers,
        names = ["hltInputLST", "hltInitialStepTrackCandidates", "hltHighPtTripletStepSeeds"])
    (~singleIterPatatrack & trackingLST & seedingLST).toModify(_seedProducers,
        names = ["hltInputLST", "hltInitialStepTrackCandidates", "hltInitialStepTrackCandidates:pLSTSsLST"])

    (_seedSelectors, _tmpTask) = _TrackValidation_cff._addSeedToTrackProducers(_seedProducers.names, globals())
    _seedSelectorsTask = cms.Task()
    for modName in _seedSelectors:
        if not hasattr(process, modName):
            setattr(process,modName, globals()[modName].clone(beamSpot = "hltOnlineBeamSpot"))
        _seedSelectorsTask.add(getattr(process, modName))

    customiseTrackingNtupleTool(process, isRECO = False)

    process.trackingNtupleSequence.insert(0,cms.Sequence(_seedSelectorsTask))
    if hasattr(process, "hltSiStripRawToClustersFacility") and process.hltSiStripRawToClustersFacility.onDemand.value():
        #make sure that all iter tracking is done before running the ntuple-related modules
        process.trackingNtupleSequence.insert(0,process.hltMergedTracks)

    process.trackingNtuple.seedTracks = _seedSelectors

    process.trackingNtuple.tracks = "hltMergedTracks"
    trackingPhase2PU140.toModify(process.trackingNtuple, tracks = "hltGeneralTracks")

    (singleIterPatatrack & trackingLST & seedingLST).toModify(process.trackingNtuple,
        seedAlgoDetect = False, seedAlgos = [getattr(_algo,"initialStep")])
    (~singleIterPatatrack & trackingLST & seedingLST).toModify(process.trackingNtuple,
        seedAlgoDetect = False, seedAlgos = [getattr(_algo,"initialStep"), getattr(_algo,"initialStep"), getattr(_algo,"highPtTripletStep")])

    process.trackingNtuple.trackCandidates = ["hltIter0PFlowCkfTrackCandidates", "hltDoubletRecoveryPFlowCkfTrackCandidates"]
    trackingPhase2PU140.toModify(process.trackingNtuple, trackCandidates = ["hltInitialStepTrackCandidates", "hltHighPtTripletStepTrackCandidates"])
    singleIterPatatrack.toModify(process.trackingNtuple, trackCandidates = ["hltInitialStepTrackCandidates"])

    process.trackingNtuple.clusterMasks = [dict(index = getattr(_algo,"pixelPairStep"), src = "hltDoubletRecoveryClustersRefRemoval")]
    trackingPhase2PU140.toModify(process.trackingNtuple, clusterMasks = [dict(index = getattr(_algo,"highPtTripletStep"), src = "hltHighPtTripletStepClusters")])
    singleIterPatatrack.toModify(process.trackingNtuple, clusterMasks = [])

    process.trackingNtuple.clusterTPMap = "hltTPClusterProducer"
    process.trackingNtuple.trackAssociator = "hltTrackAssociatorByHits"
    process.trackingNtuple.beamSpot = "hltOnlineBeamSpot"
    process.trackingNtuple.pixelRecHits = "hltSiPixelRecHits"
    process.trackingNtuple.stripRphiRecHits = "hltSiStripRecHits:rphiRecHit"
    process.trackingNtuple.stripStereoRecHits = "hltSiStripRecHits:stereoRecHit"
    process.trackingNtuple.stripMatchedRecHits = "hltSiStripRecHits:matchedRecHit"
    process.trackingNtuple.phase2OTRecHits = "hltSiPhase2RecHits"
    process.trackingNtuple.vertices = "hltPixelVertices"
    # currently not used: keep for possible future use
    process.trackingNtuple.TTRHBuilder = "hltESPTTRHBWithTrackAngle"
    trackingPhase2PU140.toModify(process.trackingNtuple, vertices = "hltPhase2PixelVertices", TTRHBuilder = "hltESPTTRHBuilderWithTrackAngle")
    process.trackingNtuple.includeMVA = False

    return process

def extendedContent(process):
    process.trackingParticlesIntime.intimeOnly = True
    shTags = [process.simHitTPAssocProducer.simHitSrc]
    if _usePileupSimHits(process):
        shTags += [process.crossingFramePSimHitToPSimHits.src]
    for vTags in shTags:
        lowT = {t for t in vTags if t.endswith("HighTof")}
        htNeeded = {t.replace("LowTof", "HighTof") for t in vTags if t.endswith("LowTof")}
        for t in (t for t in htNeeded if t not in lowT):
            vTags += [t]
    process.trackingNtuple.includeOOT = True
    process.trackingNtuple.keepEleSimHits = True

    process.trackingNtuple.saveSimHitsP3 = True
    process.trackingNtuple.addSeedCurvCov = True

    process.trackingNtuple.includeOnTrackHitData = True
    process.trackingNtuple.includeTrackCandidates = True

    return process
