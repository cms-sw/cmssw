/*
 * \file FakeTBHodoscopeRawInfoProducer.cc
 *
 * Mimic the hodoscope raw information using
 * the generated vertex of the test beam simulation
 *
 */

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
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopePlaneRawHits.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"

#include <memory>

class FakeTBHodoscopeRawInfoProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit FakeTBHodoscopeRawInfoProducer(const edm::ParameterSet &ps);

  /// Destructor
  ~FakeTBHodoscopeRawInfoProducer() override = default;

  /// Produce digis out of raw data
  void produce(edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  std::unique_ptr<EcalTBHodoscopeGeometry> theTBHodoGeom_;

  const edm::EDGetTokenT<PEcalTBInfo> ecalTBInfo_;
};

FakeTBHodoscopeRawInfoProducer::FakeTBHodoscopeRawInfoProducer(const edm::ParameterSet &ps)
    : ecalTBInfo_(consumes<PEcalTBInfo>(edm::InputTag("EcalTBInfoLabel", "SimEcalTBG4Object"))) {
  produces<EcalTBHodoscopeRawInfo>();

  theTBHodoGeom_ = std::make_unique<EcalTBHodoscopeGeometry>();
}

void FakeTBHodoscopeRawInfoProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup) {
  std::unique_ptr<EcalTBHodoscopeRawInfo> product(new EcalTBHodoscopeRawInfo());

  // get the vertex information from the event

  const edm::Handle<PEcalTBInfo> &theEcalTBInfo = event.getHandle(ecalTBInfo_);

  double partXhodo = theEcalTBInfo->evXbeam();
  double partYhodo = theEcalTBInfo->evYbeam();

  LogDebug("EcalTBHodo") << "TB frame vertex (X,Y) for hodoscope simulation: \n"
                         << "x = " << partXhodo << " y = " << partYhodo;

  // for each hodoscope plane determine the fibre number corresponding
  // to the event vertex coordinates in the TB reference frame
  // plane 0/2 = x plane 1/3 = y

  int nPlanes = static_cast<int>(theTBHodoGeom_->getNPlanes());
  product->setPlanes(nPlanes);

  for (int iPlane = 0; iPlane < nPlanes; ++iPlane) {
    float theCoord = (float)partXhodo;
    if (iPlane == 1 || iPlane == 3)
      theCoord = (float)partYhodo;

    std::vector<int> firedChannels = theTBHodoGeom_->getFiredFibresInPlane(theCoord, iPlane);
    unsigned int nChannels = firedChannels.size();

    EcalTBHodoscopePlaneRawHits planeHit(nChannels);
    for (unsigned int i = 0; i < nChannels; ++i) {
      planeHit.addHit(firedChannels[i]);
    }

    product->setPlane(static_cast<unsigned int>(iPlane), planeHit);
  }

  LogDebug("EcalTBHodo") << (*product);

  event.put(std::move(product));
}

DEFINE_FWK_MODULE(FakeTBHodoscopeRawInfoProducer);
