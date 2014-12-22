import FWCore.ParameterSet.Config as cms

def customiseSequentialSim(process):

    for label, prod in process.producers_().iteritems():
        if prod.type_() == "OscarMTProducer":
            # ugly hack
            prod.__dict__['_TypedParameterizable__type'] = "OscarProducer"

    return process

