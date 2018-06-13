import FWCore.ParameterSet.Config as cms
import six

def customiseSequentialSim(process):
    # Set numberOfStreams to allow cmsRun/cmsDriver.py -n to control
    # also the number of streams

    for label, prod in process.producers_(six.iteritems()):
        if prod.type_() == "OscarMTProducer":
            # ugly hack
            prod.__dict__['_TypedParameterizable__type'] = "OscarProducer"

    return process

