#ifndef SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimSpikeESProducer_H
#define SimCalorimetry_EcalTrigPrimProducers_EcalTrigPrimSpikeESProducer_H

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"

#include <vector>

//
// class declaration
//

class EcalTrigPrimSpikeESProducer : public edm::ESProducer {
 public:
  EcalTrigPrimSpikeESProducer(const edm::ParameterSet&);
  ~EcalTrigPrimSpikeESProducer();

  std::auto_ptr<EcalTPGSpike> produceSpike(const EcalTPGSpikeRcd &) ;

 private:
  std::vector<uint32_t> towerIDs_;
  uint16_t zeroThresh_;
};


#endif
