#ifndef ME0Digitizer_ME0DigiModelFactory_h
#define ME0Digitizer_ME0DigiModelFactory_h

/** \class ME0DigiModelFactory
 *
 * Factory of seal plugins for ME0Digitizer
 *
 * \author Vadim Khotilovich
 *
 */
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm{
  class ParameterSet;
}

class ME0DigiModel;

typedef edmplugin::PluginFactory<ME0DigiModel *(const edm::ParameterSet &)> ME0DigiModelFactory;

#endif
