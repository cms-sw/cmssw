#include "SimCalorimetry/HcalSimProducers/plugins/HcalDigiProducer.h"
using namespace std;


HcalDigiProducer::HcalDigiProducer(const edm::ParameterSet& ps) 
: theDigitizer(ps)
{
  produces<HBHEDigiCollection>();
  produces<HODigiCollection>();
  produces<HFDigiCollection>();
  produces<ZDCDigiCollection>();
  produces<HcalUpgradeDigiCollection>();
}


HcalDigiProducer::~HcalDigiProducer() {
}


void HcalDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  theDigitizer.produce(e, eventSetup);
}

