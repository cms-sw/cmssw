#ifndef TrivialSerialisation_src_TrivialSerialiserSourceFactory_h
#define TrivialSerialisation_src_TrivialSerialiserSourceFactory_h

#include "TrivialSerialisation/Common/interface/TrivialSerialiserSourceBase.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace ngt {
  using TrivialSerialiserSourceFactory = edmplugin::PluginFactory<ngt::TrivialSerialiserSourceBase *()>;
}
#endif  // TrivialSerialisation_src_TrivialSerialiserSourceFactory_h
