/*
 * \file TBHodoActiveVolumeRawInfoProducer.cc
 *
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
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopePlaneRawHits.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"

#include <iostream>
#include <memory>

class TBHodoActiveVolumeRawInfoProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit TBHodoActiveVolumeRawInfoProducer(const edm::ParameterSet &ps);

  /// Destructor
  ~TBHodoActiveVolumeRawInfoProducer() override = default;

  /// Produce digis out of raw data
  void produce(edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  double myThreshold;
  edm::EDGetTokenT<edm::PCaloHitContainer> m_EcalToken;

  std::unique_ptr<EcalTBHodoscopeGeometry> theTBHodoGeom_;
};

using namespace cms;
using namespace std;

TBHodoActiveVolumeRawInfoProducer::TBHodoActiveVolumeRawInfoProducer(const edm::ParameterSet &ps)
    : myThreshold(0.05E-3),
      m_EcalToken(consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalTBH4BeamHits"))) {
  produces<EcalTBHodoscopeRawInfo>();

  theTBHodoGeom_ = std::make_unique<EcalTBHodoscopeGeometry>();
}

void TBHodoActiveVolumeRawInfoProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup) {
  std::unique_ptr<EcalTBHodoscopeRawInfo> product(new EcalTBHodoscopeRawInfo());

  // caloHit container
  const edm::Handle<edm::PCaloHitContainer> &pCaloHit = event.getHandle(m_EcalToken);
  const edm::PCaloHitContainer *caloHits = nullptr;
  if (pCaloHit.isValid()) {
    caloHits = pCaloHit.product();
    LogDebug("EcalTBHodo") << "total # caloHits: " << caloHits->size();
  } else {
    edm::LogError("EcalTBHodo") << "Error! can't get the caloHitContainer ";
  }
  if (!caloHits) {
    return;
  }

  // detid - energy_sum map
  std::map<unsigned int, double> energyMap;

  for (auto &&aHit : *caloHits) {
    double thisHitEne = aHit.energy();

    std::map<unsigned int, double>::iterator itmap = energyMap.find(aHit.id());
    if (itmap == energyMap.end())
      energyMap.insert(pair<unsigned int, double>(aHit.id(), thisHitEne));
    else {
      (*itmap).second += thisHitEne;
    }
  }

  // planes and fibers
  int nPlanes = theTBHodoGeom_->getNPlanes();
  int nFibers = theTBHodoGeom_->getNFibres();
  product->setPlanes(nPlanes);

  bool firedChannels[4][64];
  for (int iPlane = 0; iPlane < nPlanes; ++iPlane) {
    for (int iFiber = 0; iFiber < nFibers; ++iFiber) {
      firedChannels[iPlane][iFiber] = 0.;
    }
  }
  for (std::map<unsigned int, double>::const_iterator itmap = energyMap.begin(); itmap != energyMap.end(); ++itmap) {
    if ((*itmap).second > myThreshold) {
      HodoscopeDetId myHodoDetId = HodoscopeDetId((*itmap).first);
      firedChannels[myHodoDetId.planeId()][myHodoDetId.fibrId()] = true;
    }
  }
  for (int iPlane = 0; iPlane < nPlanes; ++iPlane) {
    EcalTBHodoscopePlaneRawHits planeHit(nFibers);

    for (int iFiber = 0; iFiber < nFibers; ++iFiber) {
      planeHit.setHit(iFiber, firedChannels[iPlane][iFiber]);
    }
    product->setPlane((unsigned int)iPlane, planeHit);
  }

  LogDebug("EcalTBHodo") << (*product);

  event.put(std::move(product));
}

DEFINE_FWK_MODULE(TBHodoActiveVolumeRawInfoProducer);
