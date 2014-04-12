#ifndef TrackingTools_TrackAssociator_DetIdAssociatorFactory_h
#define TrackingTools_TrackAssociator_DetIdAssociatorFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "TrackingTools/TrackAssociator/interface/DetIdAssociator.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

typedef edmplugin::PluginFactory<DetIdAssociator * ( const edm::ParameterSet& pSet )> DetIdAssociatorFactory;

#endif
