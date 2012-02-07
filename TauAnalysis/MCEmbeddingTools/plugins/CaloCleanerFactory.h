#ifndef CaloCleanerFactory_h 
#define CaloCleanerFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm{
  class ParameterSet;
}

class CaloCleanerBase;

typedef edmplugin::PluginFactory<CaloCleanerBase *(const edm::ParameterSet &)> CaloCleanerFactory;

#endif

