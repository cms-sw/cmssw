#ifndef SimGeneral_PreMixingModule_PreMixingWorkerFactory_h
#define SimGeneral_PreMixingModule_PreMixingWorkerFactory_h

#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"

namespace edm {
  class ParameterSet;
  class ConsumesCollector;
}  // namespace edm

using PreMixingWorkerFactory = edmplugin::PluginFactory<PreMixingWorker*(
    const edm::ParameterSet&, edm::ProducesCollector, edm::ConsumesCollector&& iC)>;

#define DEFINE_PREMIXING_WORKER(TYPE) DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, TYPE, #TYPE)

#endif
