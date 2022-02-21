/*
 * \file FakeTBEventHeaderProducer.cc
 *
 * Mimic the event header information
 * for the test beam simulation
 *
 */

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

class FakeTBEventHeaderProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit FakeTBEventHeaderProducer(const edm::ParameterSet &ps);

  /// Destructor
  ~FakeTBEventHeaderProducer() override = default;

  /// Produce digis out of raw data
  void produce(edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  const edm::EDGetTokenT<PEcalTBInfo> ecalTBInfo_;
};

FakeTBEventHeaderProducer::FakeTBEventHeaderProducer(const edm::ParameterSet &ps)
    : ecalTBInfo_(consumes<PEcalTBInfo>(edm::InputTag("EcalTBInfoLabel", "SimEcalTBG4Object"))) {
  produces<EcalTBEventHeader>();
}

void FakeTBEventHeaderProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup) {
  std::unique_ptr<EcalTBEventHeader> product(new EcalTBEventHeader());

  // get the vertex information from the event

  const PEcalTBInfo *theEcalTBInfo = nullptr;
  const edm::Handle<PEcalTBInfo> &EcalTBInfo = event.getHandle(ecalTBInfo_);
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

  LogDebug("FakeTBHeader") << (*product);
  LogDebug("FakeTBHeader") << (*product).eventType();
  LogDebug("FakeTBHeader") << (*product).crystalInBeam();

  event.put(std::move(product));
}

DEFINE_FWK_MODULE(FakeTBEventHeaderProducer);
