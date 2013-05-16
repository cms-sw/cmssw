#ifndef GEMDigitizer_GEMFactory_h
#define GEMDigitizer_GEMFactory_h

/** \class GEMFactory
 *
 * Factory of seal plugins for GEMDigitizer
 *
 * \author Sven Dildick
 *
 */

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

namespace edm{
  class ParameterSet;
}

class GEMTiming;
class GEMNoise; 
class GEMClustering; 
class GEMEfficiency; 

typedef edmplugin::PluginFactory<GEMTiming *(const edm::ParameterSet &)> GEMTimingFactory;
typedef edmplugin::PluginFactory<GEMNoise *(const edm::ParameterSet &)> GEMNoiseFactory; 
typedef edmplugin::PluginFactory<GEMClustering *(const edm::ParameterSet &)> GEMClusteringFactory; 
typedef edmplugin::PluginFactory<GEMEfficiency *(const edm::ParameterSet &)> GEMEfficiencyFactory; 

#endif
