import FWCore.ParameterSet.Config as cms
import six

def _label(tag):
    if hasattr(tag, "getModuleLabel"):
        t = tag
    else:
        t = cms.InputTag(tag)
    return t.getModuleLabel()+t.getProductInstanceLabel()

def customiseTrackingNtuple(process):
    process.load("Validation.RecoTrack.trackingNtuple_cff")
    process.TFileService = cms.Service("TFileService",
        fileName = cms.string('trackingNtuple.root')
    )

    if process.trackingNtuple.includeSeeds.value():
        if not hasattr(process, "reconstruction_step"):
            raise Exception("TrackingNtuple includeSeeds=True needs reconstruction which is missing")

    # Replace validation_step with ntuplePath
    if not hasattr(process, "validation_step"):
        raise Exception("TrackingNtuple customise assumes process.validation_step exists")


    # Should replay mixing for pileup simhits?
    usePileupSimHits = hasattr(process, "mix") and hasattr(process.mix, "input") and len(process.mix.input.fileNames) > 0
#    process.eda = cms.EDAnalyzer("EventContentAnalyzer")

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
