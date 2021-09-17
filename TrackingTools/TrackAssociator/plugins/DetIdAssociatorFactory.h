#ifndef TrackingTools_TrackAssociator_DetIdAssociatorFactory_h
#define TrackingTools_TrackAssociator_DetIdAssociatorFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "DetIdAssociatorMaker.h"

typedef edmplugin::PluginFactory<DetIdAssociatorMaker*(const edm::ParameterSet& pSet,
                                                       edm::ESConsumesCollectorT<DetIdAssociatorRecord>&&)>
    DetIdAssociatorFactory;

#endif
