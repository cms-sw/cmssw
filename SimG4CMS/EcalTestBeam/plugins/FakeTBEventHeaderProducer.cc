/*
 * \file FakeTBEventHeaderProducer.cc
 *
 *
 */

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "SimG4CMS/EcalTestBeam/interface/FakeTBEventHeaderProducer.h"

using namespace cms;
using namespace std;

FakeTBEventHeaderProducer::FakeTBEventHeaderProducer(const edm::ParameterSet &ps) {
  ecalTBInfo_ = consumes<PEcalTBInfo>(edm::InputTag("EcalTBInfoLabel", "SimEcalTBG4Object"));
  produces<EcalTBEventHeader>();
}

FakeTBEventHeaderProducer::~FakeTBEventHeaderProducer() {}

void FakeTBEventHeaderProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup) {
  unique_ptr<EcalTBEventHeader> product(new EcalTBEventHeader());

  // get the vertex information from the event

  const PEcalTBInfo *theEcalTBInfo = nullptr;
  edm::Handle<PEcalTBInfo> EcalTBInfo;
  event.getByToken(ecalTBInfo_, EcalTBInfo);
  if (EcalTBInfo.isValid()) {
    theEcalTBInfo = EcalTBInfo.product();
  } else {
    edm::LogError("FakeTBEventHeaderProducer") << "Error! can't get the product PEcalTBInfo";
  }

  if (!theEcalTBInfo) {
    return;
  }

  // 64 bits event ID in CMSSW converted to EcalTBEventHeader ID
  int evtid = (int)event.id().event();
  product->setEventNumber(evtid);
  product->setRunNumber(event.id().run());
  product->setBurstNumber(1);
  product->setTriggerMask(0x1);
  product->setCrystalInBeam(EBDetId(1, theEcalTBInfo->nCrystal(), EBDetId::SMCRYSTALMODE));

  //   LogDebug("FakeTBHeader") << (*product);
  //   LogDebug("FakeTBHeader") << (*product).eventType();
  //   LogDebug("FakeTBHeader") << (*product).crystalInBeam();
  event.put(std::move(product));
}
