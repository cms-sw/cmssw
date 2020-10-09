#include <memory>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/EcalCATIAGainRatiosRcd.h"
#include "CondFormats/EcalObjects/interface/EcalCATIAGainRatios.h"
#include "CondFormats/EcalObjects/src/classes.h"
//
// class declaration
//
const int kEBChannels = ecalPh2::kEBChannels;

class EcalCATIAGainRatiosESProducer : public edm::ESProducer {
public:
  EcalCATIAGainRatiosESProducer(const edm::ParameterSet& p);

  typedef std::unique_ptr<EcalCATIAGainRatios> ReturnType;

  ReturnType produce(const EcalCATIAGainRatiosRcd& iRecord);

private:
  double catiaGainRatio_;
};

using namespace edm;

EcalCATIAGainRatiosESProducer::EcalCATIAGainRatiosESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  catiaGainRatio_ = p.getParameter<double>("CATIAGainRatio");
  setWhatProduced(this);
}
////
EcalCATIAGainRatiosESProducer::ReturnType EcalCATIAGainRatiosESProducer::produce(const EcalCATIAGainRatiosRcd& iRecord) {
  auto prod = std::make_unique<EcalCATIAGainRatios>();
  for (int iChannel = 0; iChannel < kEBChannels; iChannel++) {
    EBDetId myEBDetId = EBDetId::unhashIndex(iChannel);
    double val = catiaGainRatio_;
    prod->setValue(myEBDetId.rawId(), val);
  }

  return prod;
}

//Define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalCATIAGainRatiosESProducer);
