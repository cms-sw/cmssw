#ifndef Utilities_StorageFactory_StorageProxyMakerFactory_h
#define Utilities_StorageFactory_StorageProxyMakerFactory_h

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include <memory>

namespace edm::storage {
  class StorageProxyMaker;
  using StorageProxyMakerFactory = edmplugin::PluginFactory<StorageProxyMaker*(edm::ParameterSet const&)>;
};  // namespace edm::storage

#endif
