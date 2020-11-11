#include <memory>
#include <string>
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESProductHost.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/DataRecord/interface/EcalLiteDTUPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalLiteDTUPedestals.h"
#include "CondFormats/EcalObjects/src/classes.h"
#include "DataFormats/EcalDigi/interface/EcalConstants.h"

//
// class declaration
//

class EcalLiteDTUPedestalsESProducer : public edm::ESProducer {
public:
  EcalLiteDTUPedestalsESProducer(const edm::ParameterSet& p);

  typedef std::unique_ptr<EcalLiteDTUPedestalsMap> ReturnType;

  ReturnType produce(const EcalLiteDTUPedestalsRcd& iRecord);
  //Add 2 nov 2020:
  edm::ESGetToken<EcalLiteDTUPedestalsMap, EcalLiteDTUPedestalsRcd> pedestalToken_;
  ///////////////////////////////
private:
  double meanPedestalsGain10_;
  double rmsPedestalsGain10_;
  double meanPedestalsGain1_;
  double rmsPedestalsGain1_;
};

using namespace edm;

EcalLiteDTUPedestalsESProducer::EcalLiteDTUPedestalsESProducer(const edm::ParameterSet& p) {
  std::string myname = p.getParameter<std::string>("ComponentName");
  meanPedestalsGain10_ = p.getParameter<double>("MeanPedestalsGain10");
  rmsPedestalsGain10_ = p.getParameter<double>("RMSPedestalsGain10");
  meanPedestalsGain1_ = p.getParameter<double>("MeanPedestalsGain1");
  rmsPedestalsGain1_ = p.getParameter<double>("RMSPedestalsGain1");
  auto cc = setWhatProduced(this);
  pedestalToken_ = cc.consumes<EcalLiteDTUPedestalsMap>();
}
////
EcalLiteDTUPedestalsESProducer::ReturnType EcalLiteDTUPedestalsESProducer::produce(
    const EcalLiteDTUPedestalsRcd& iRecord) {
  auto prod = std::make_unique<EcalLiteDTUPedestalsMap>();

  for (unsigned int iChannel = 0; iChannel < ecalPh2::kEBChannels; iChannel++) {
    EBDetId myEBDetId = EBDetId::unhashIndex(iChannel);
    EcalLiteDTUPedestals ped;
    ped.setMean(0, meanPedestalsGain10_);
    ped.setRMS(0, rmsPedestalsGain10_);

    ped.setMean(1, meanPedestalsGain1_);
    ped.setRMS(1, rmsPedestalsGain1_);

    prod->insert(std::make_pair(myEBDetId, ped));
  }

  return prod;
}

//Define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(EcalLiteDTUPedestalsESProducer);
