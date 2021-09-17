#ifndef SimCalorimetry_HGCalSimProducers_HGCDigitizerPluginFactory_H
#define SimCalorimetry_HGCalSimProducers_HGCDigitizerPluginFactory_H

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/HGCalSimProducers/interface/HGCDigitizerBase.h"

typedef edmplugin::PluginFactory<HGCDigitizerBase*(const edm::ParameterSet&)> HGCDigitizerPluginFactory;

#endif
