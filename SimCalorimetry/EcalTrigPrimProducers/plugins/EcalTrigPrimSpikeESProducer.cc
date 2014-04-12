// user include files
#include "EcalTrigPrimSpikeESProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

//
// constructors and destructor
//
EcalTrigPrimSpikeESProducer::EcalTrigPrimSpikeESProducer(const edm::ParameterSet& iConfig) :
  zeroThresh_(iConfig.getUntrackedParameter<uint32_t>("TCCZeroingThreshold", 0))
{
  // Indicate we produce the spike record
  setWhatProduced(this, &EcalTrigPrimSpikeESProducer::produceSpike) ;

  // Cache all EB TT raw DetIDs
  for(unsigned int i = 1; i <= 17; ++i)
  {
    for(unsigned int j = 1; j <= 72; ++j)
    {
      EcalTrigTowerDetId posTT(1, EcalBarrel, i, j);
      EcalTrigTowerDetId negTT(-1, EcalBarrel, i, j);
      towerIDs_.push_back(posTT.rawId());
      towerIDs_.push_back(negTT.rawId());
    }
  }
}

EcalTrigPrimSpikeESProducer::~EcalTrigPrimSpikeESProducer()
{ 
}


// ------------ method called to produce the data  ------------
std::auto_ptr<EcalTPGSpike> EcalTrigPrimSpikeESProducer::produceSpike(const EcalTPGSpikeRcd &iRecord)
{
  std::auto_ptr<EcalTPGSpike> prod(new EcalTPGSpike());
  for(std::vector<uint32_t>::const_iterator it = towerIDs_.begin(); it != towerIDs_.end(); ++it)
  {
    prod->setValue(*it, zeroThresh_);
  }
  return prod;
}

