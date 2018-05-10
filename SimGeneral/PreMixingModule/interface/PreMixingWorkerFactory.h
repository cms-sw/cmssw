#ifndef SimGeneral_PreMixingModule_PreMixingWorkerFactory_h
#define SimGeneral_PreMixingModule_PreMixingWorkerFactory_h

#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}

using PreMixingWorkerFactory = edmplugin::PluginFactory< PreMixingWorker* (const edm::ParameterSet&, edm::ProducerBase&, edm::ConsumesCollector&& iC) >;

#endif
