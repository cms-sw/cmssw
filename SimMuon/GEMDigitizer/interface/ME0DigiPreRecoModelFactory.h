#ifndef GEMDigitizer_ME0DigiPreRecoModelFactory_h
#define GEMDigitizer_ME0DigiPreRecoModelFactory_h

/** \class ME0DigiPreRecoModelFactory
 *
 * Factory of seal plugins for ME0PreRecoDigitizer
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm{
  class ParameterSet;
}

class ME0DigiPreRecoModel;

typedef edmplugin::PluginFactory<ME0DigiPreRecoModel *(const edm::ParameterSet &)> ME0DigiPreRecoModelFactory;

#endif
