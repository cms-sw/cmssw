#ifndef SimGeneral_MixingModule_DigiAccumulatorMixModFactory_h
#define SimGeneral_MixingModule_DigiAccumulatorMixModFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "FWCore/Framework/interface/ProducerBase.h"

namespace edm {
  class ConsumesCollector;
  class ParameterSet;

  typedef DigiAccumulatorMixMod*(DAFunc)(ParameterSet const&, ProducerBase&, ConsumesCollector&);
  typedef edmplugin::PluginFactory<DAFunc> DigiAccumulatorMixModPluginFactory;

  class DigiAccumulatorMixModFactory {
  public:
    ~DigiAccumulatorMixModFactory();

    static DigiAccumulatorMixModFactory const* get();

    std::unique_ptr<DigiAccumulatorMixMod> makeDigiAccumulator(ParameterSet const&,
                                                               ProducerBase&,
                                                               ConsumesCollector&) const;

  private:
    DigiAccumulatorMixModFactory();
    static DigiAccumulatorMixModFactory const singleInstance_;
  };
}  // namespace edm

#define DEFINE_DIGI_ACCUMULATOR(type) DEFINE_EDM_PLUGIN(edm::DigiAccumulatorMixModPluginFactory, type, #type)
//DEFINE_EDM_PLUGIN (edm::DigiAccumulatorMixModPluginFactory,type,#type); DEFINE_FWK_PSET_DESC_FILLER(type)

#endif
