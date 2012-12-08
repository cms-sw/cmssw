#ifndef GEMDigitizer_GEMSimFactory_h
#define GEMDigitizer_GEMSimFactory_h

/** \class GEMSimFactory
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

class GEMSim;

typedef edmplugin::PluginFactory<GEMSim *(const edm::ParameterSet &)> GEMSimFactory;

#endif
