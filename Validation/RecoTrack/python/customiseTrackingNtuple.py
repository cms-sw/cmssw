import FWCore.ParameterSet.Config as cms

def customiseTrackingNtuple(process):
    process.load("Validation.RecoTrack.trackingNtuple_cff")
    process.TFileService = cms.Service("TFileService",
        fileName = cms.string('trackingNtuple.root')
    )

    if process.trackingNtuple.includeSeeds.value() and not hasattr(process, "reconstruction_step"):
        raise Exception("TrackingNtuple includeSeeds=True needs reconstruction which is missing")

    # Replace validation_step with ntuplePath
    if not hasattr(process, "validation_step"):
        raise Exception("TrackingNtuple customise assumes process.validation_step exists")

    # Bit of a hack but works
    modifier = cms.Modifier()
    modifier._setChosen()
    modifier.toReplaceWith(process.validation_step, cms.EndPath(process.trackingNtupleSequence))

    if hasattr(process, "prevalidation_step"):
        modifier.toReplaceWith(process.prevalidation_step, cms.Path())

    # Remove all output modules
    for outputModule in process.outputModules_().itervalues():
        for path in process.paths_().itervalues():
            path.remove(outputModule)
        for path in process.endpaths_().itervalues():
            path.remove(outputModule)
        

    return process
