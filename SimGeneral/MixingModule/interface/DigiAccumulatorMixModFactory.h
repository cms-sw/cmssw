#ifndef SimGeneral_MixingModule_DigiAccumulatorMixModFactory_h
#define SimGeneral_MixingModule_DigiAccumulatorMixModFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"

namespace edm {
  class ConsumesCollector;
  class EDProducer;
  class ParameterSet;

  typedef DigiAccumulatorMixMod*(DAFunc)(ParameterSet const&, EDProducer&, ConsumesCollector&);
  typedef edmplugin::PluginFactory<DAFunc> DigiAccumulatorMixModPluginFactory;

  class DigiAccumulatorMixModFactory {
  public:
    ~DigiAccumulatorMixModFactory();

    static DigiAccumulatorMixModFactory* get();

    std::auto_ptr<DigiAccumulatorMixMod>
      makeDigiAccumulator(ParameterSet const&, EDProducer&, ConsumesCollector&) const;

  private:
    DigiAccumulatorMixModFactory();
    static DigiAccumulatorMixModFactory singleInstance_;
  };
}

#define DEFINE_DIGI_ACCUMULATOR(type) \
  DEFINE_EDM_PLUGIN (edm::DigiAccumulatorMixModPluginFactory,type,#type)
  //DEFINE_EDM_PLUGIN (edm::DigiAccumulatorMixModPluginFactory,type,#type); DEFINE_FWK_PSET_DESC_FILLER(type)

#endif

