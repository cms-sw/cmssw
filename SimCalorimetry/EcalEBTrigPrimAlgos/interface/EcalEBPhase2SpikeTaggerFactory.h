#ifndef SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2SpikeTaggerFactory_h
#define SimCalorimetry_EcalEBTrigPrimAlgos_EcalEBPhase2SpikeTaggerFactory_h

#include <memory>

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBPhase2SpikeTagger.h"

class EcalEBPhase2SpikeTaggerFactory {
public:
  EcalEBPhase2SpikeTaggerFactory() {};
  ~EcalEBPhase2SpikeTaggerFactory() {};

  typedef std::unique_ptr<EcalEBPhase2SpikeTagger> ReturnType;

  ReturnType create(std::string const& algoType, uint32_t version, edm::ConsumesCollector& cc, bool debug);
};

#endif
