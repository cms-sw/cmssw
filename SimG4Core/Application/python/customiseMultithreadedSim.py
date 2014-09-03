import FWCore.ParameterSet.Config as cms

def customiseMultithreadedSim(process):
    # Set numberOfStreams to allow cmsRun/cmsDriver.py -n to control
    # also the number of streams
    if not hasattr(process, "options"):
        process.options = cms.PSet()
    if not hasattr(process.options, "numberOfStreams"):
        process.options.numberOfStreams = cms.untracked.uint32(0)

    for label, prod in process.producers_().iteritems():
        if prod.type_() == "OscarProducer":
            # ugly hack
            prod.__dict__['_TypedParameterizable__type'] = "OscarMTProducer"

    return process

