
#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixModFactory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/DebugMacros.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iostream>

EDM_REGISTER_PLUGINFACTORY(edm::DigiAccumulatorMixModPluginFactory, "DigiAccumulator");

namespace edm {
  DigiAccumulatorMixModFactory::~DigiAccumulatorMixModFactory() {}

  DigiAccumulatorMixModFactory::DigiAccumulatorMixModFactory() {}

  DigiAccumulatorMixModFactory const DigiAccumulatorMixModFactory::singleInstance_;

  DigiAccumulatorMixModFactory const* DigiAccumulatorMixModFactory::get() {
    // will not work with plugin factories
    //static DigiAccumulatorMixModFactory f;
    //return &f;

    return &singleInstance_;
  }

  std::unique_ptr<DigiAccumulatorMixMod> DigiAccumulatorMixModFactory::makeDigiAccumulator(
      ParameterSet const& conf, ProducesCollector producesCollector, ConsumesCollector& iC) const {
    std::string accumulatorType = conf.getParameter<std::string>("accumulatorType");
    FDEBUG(1) << "DigiAccumulatorMixModFactory: digi_accumulator_type = " << accumulatorType << std::endl;
    auto wm = DigiAccumulatorMixModPluginFactory::get()->create(accumulatorType, conf, producesCollector, iC);

    if (wm.get() == nullptr) {
      throw edm::Exception(errors::Configuration, "NoSourceModule")
          << "DigiAccumulator Factory:\n"
          << "Cannot find dig type from ParameterSet: " << accumulatorType << "\n"
          << "Perhaps your source type is misspelled or is not an EDM Plugin?\n"
          << "Try running EdmPluginDump to obtain a list of available Plugins.";
    }

    FDEBUG(1) << "DigiAccumulatorMixModFactory: created a Digi Accumulator " << accumulatorType << std::endl;

    return wm;
  }
}  // namespace edm
