#include "SimCalorimetry/HcalSimProducers/plugins/HcalDigiProducer.h"
using namespace std;


HcalDigiProducer::HcalDigiProducer(const edm::ParameterSet& ps) 
: theDigitizer(ps)
{
  produces<HBHEDigiCollection>();
  produces<HODigiCollection>();
  produces<HFDigiCollection>();
  produces<ZDCDigiCollection>();
}


HcalDigiProducer::~HcalDigiProducer() {
}


void HcalDigiProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  theDigitizer.produce(e, eventSetup);
}

void HcalDigiProducer::beginRun(edm::Run& run, edm::EventSetup const& es)
{
  theDigitizer.beginRun(es);
}

void HcalDigiProducer::endRun(edm::Run& run, edm::EventSetup const& es)
{
  theDigitizer.endRun();
}

