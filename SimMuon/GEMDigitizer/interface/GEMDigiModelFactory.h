#ifndef GEMDigitizer_GEMDigiModelFactory_h
#define GEMDigitizer_GEMDigiModelFactory_h

/** \class GEMDigiModelFactory
 *
 * Factory of seal plugins for GEMDigitizer
 *
 * \author Vadim Khotilovich
 *
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm{
  class ParameterSet;
}

class GEMDigiModel;

typedef edmplugin::PluginFactory<GEMDigiModel *(const edm::ParameterSet &)> GEMDigiModelFactory;

#endif
