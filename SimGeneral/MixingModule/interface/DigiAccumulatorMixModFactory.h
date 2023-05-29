#ifndef SimGeneral_MixingModule_DigiAccumulatorMixModFactory_h
#define SimGeneral_MixingModule_DigiAccumulatorMixModFactory_h

#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"

namespace edm {
  class ConsumesCollector;
  class ParameterSet;

  typedef DigiAccumulatorMixMod*(DAFunc)(ParameterSet const&, ProducesCollector, ConsumesCollector&);
  typedef edmplugin::PluginFactory<DAFunc> DigiAccumulatorMixModPluginFactory;

  class DigiAccumulatorMixModFactory {
  public:
    ~DigiAccumulatorMixModFactory();

    static DigiAccumulatorMixModFactory const* get();

    std::unique_ptr<DigiAccumulatorMixMod> makeDigiAccumulator(ParameterSet const&,
                                                               ProducesCollector,
                                                               ConsumesCollector&) const;

  private:
    DigiAccumulatorMixModFactory();
    static DigiAccumulatorMixModFactory const singleInstance_;
  };
}  // namespace edm

#define DEFINE_DIGI_ACCUMULATOR(type) DEFINE_EDM_PLUGIN(edm::DigiAccumulatorMixModPluginFactory, type, #type)

#endif
