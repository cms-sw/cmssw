/*
 * \file TBHodoActiveVolumeRawInfoProducer.cc
 *
 *
 */

#include "SimG4CMS/EcalTestBeam/interface/TBHodoActiveVolumeRawInfoProducer.h"

#include <iostream>

using namespace cms;
using namespace std;

TBHodoActiveVolumeRawInfoProducer::TBHodoActiveVolumeRawInfoProducer(const edm::ParameterSet &ps) {
  m_EcalToken = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalTBH4BeamHits"));
  produces<EcalTBHodoscopeRawInfo>();

  theTBHodoGeom_ = new EcalTBHodoscopeGeometry();

  myThreshold = 0.05E-3;
}

TBHodoActiveVolumeRawInfoProducer::~TBHodoActiveVolumeRawInfoProducer() { delete theTBHodoGeom_; }

void TBHodoActiveVolumeRawInfoProducer::produce(edm::Event &event, const edm::EventSetup &eventSetup) {
  unique_ptr<EcalTBHodoscopeRawInfo> product(new EcalTBHodoscopeRawInfo());

  // caloHit container
  edm::Handle<edm::PCaloHitContainer> pCaloHit;
  const edm::PCaloHitContainer *caloHits = nullptr;
  event.getByToken(m_EcalToken, pCaloHit);
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
