#ifndef HcalDummyHitProducer_h
#define HcalDummyHitProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "Geometry/HcalTowerAlgo/interface/HcalTrigTowerGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalTrigTowerDetId.h"
#include <map>


class HcalDummyHitProducer : public edm::EDProducer
{
public:
  
  typedef std::multimap<HcalTrigTowerDetId, HcalDetId> tid2cid;
  explicit HcalDummyHitProducer(const edm::ParameterSet& ps);
  virtual ~HcalDummyHitProducer();

  /**Produces the EDM products,*/
  virtual void produce(edm::Event& e, const edm::EventSetup& c);

private:
  tid2cid Tower_map;
  HcalTrigTowerGeometry theTrigTowerGeometry;
  double energyEM;
  double energyHad;
  double step_size;
};

#endif

