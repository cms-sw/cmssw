#ifndef TrivialSerialisation_src_SerialiserFactory_h
#define TrivialSerialisation_src_SerialiserFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "TrivialSerialisation/Common/interface/Serialiser.h"
#include "TrivialSerialisation/Common/interface/SerialiserBase.h"

namespace ngt {
  using SerialiserFactory = edmplugin::PluginFactory<ngt::SerialiserBase*()>;
}

// Helper macro to define Serialiser plugins
#define DEFINE_TRIVIAL_SERIALISER_PLUGIN(TYPE) \
  DEFINE_EDM_PLUGIN(ngt::SerialiserFactory, ngt::Serialiser<TYPE>, typeid(TYPE).name())

#endif  // TrivialSerialisation_src_SerialiserFactory_h
