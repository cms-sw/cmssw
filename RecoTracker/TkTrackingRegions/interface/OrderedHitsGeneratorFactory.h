#ifndef RecoTracker_TkTrackingRegions_OrderedHitsGeneratorFactory_H
#define RecoTracker_TkTrackingRegions_OrderedHitsGeneratorFactory_H

#include "RecoTracker/TkTrackingRegions/interface/OrderedHitsGenerator.h"
namespace edm {class ParameterSet;}

#include "FWCore/PluginManager/interface/PluginFactory.h"

class OrderedHitsGeneratorFactory 
   : public seal::PluginFactory< OrderedHitsGenerator * (const edm::ParameterSet&) > {
public:
  OrderedHitsGeneratorFactory();
  virtual ~OrderedHitsGeneratorFactory();
  static OrderedHitsGeneratorFactory * get();
};
#endif

