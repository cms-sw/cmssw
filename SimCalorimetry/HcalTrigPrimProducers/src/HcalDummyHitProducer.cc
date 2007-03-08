using namespace std;
#include "SimCalorimetry/HcalTrigPrimProducers/src/HcalDummyHitProducer.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "CalibFormats/HcalObjects/interface/HcalDbRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVSimParameterMap.h"

HcalDummyHitProducer::HcalDummyHitProducer(const edm::ParameterSet& ps)
  : theParameterMap(new HcalSimParameterMap())   
{
  energy = 0;
  produces<edm::PCaloHitContainer>("HcalHits");
  step_size = ps.getParameter<double>("step_size");
}

HcalDummyHitProducer::~HcalDummyHitProducer() {}


void HcalDummyHitProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  //edm::Handle<edm::PCaloHitContainer> pcalo;
  //e.getByLabel("g4SimHits","HcalHits",pcalo);
  // get the appropriate gains, noises, & widths for this event
  edm::ESHandle<HcalDbService> conditions;
  eventSetup.get<HcalDbRecord>().get(conditions);
  std::auto_ptr<edm::PCaloHitContainer> result(new edm::PCaloHitContainer);
  // get the correct geometry
  edm::ESHandle<CaloGeometry> geometry;
  eventSetup.get<IdealGeometryRecord>().get(geometry);
  vector<DetId> hbCells =  geometry->getValidDetIds(DetId::Hcal, HcalBarrel);
  vector<DetId> heCells =  geometry->getValidDetIds(DetId::Hcal, HcalEndcap);
  vector<DetId> hfCells =  geometry->getValidDetIds(DetId::Hcal, HcalForward);

  vector<HcalTrigTowerDetId> towerids;
  Tower_map.clear();
  for(vector<DetId>::const_iterator itr = hbCells.begin(); itr != hbCells.end(); ++itr)
    {
      HcalDetId barrelId(*itr);
      towerids = theTrigTowerGeometry.towerIds(barrelId);
      for(unsigned int n = 0; n < towerids.size(); ++n)
	{
	  Tower_map.insert(tid2cid::value_type(towerids[n],barrelId));
	}
    }
  for(vector<DetId>::const_iterator itr = heCells.begin(); itr != heCells.end(); ++itr)
    {      
      HcalDetId endcapId(*itr);
      towerids = theTrigTowerGeometry.towerIds(endcapId);
      for(unsigned int n = 0; n < towerids.size(); ++n)
      {
        Tower_map.insert(tid2cid::value_type(towerids[n],endcapId));
      }
    }
  for(vector<DetId>::const_iterator itr = hfCells.begin(); itr != hfCells.end(); ++itr)
    {
      HcalDetId forwardId(*itr);
      towerids = theTrigTowerGeometry.towerIds(forwardId);
      for(unsigned int n = 0; n < towerids.size(); ++n)
      {
        Tower_map.insert(tid2cid::value_type(towerids[n],forwardId));
      }
    }

  double time;
  double calib;
  int ncells;
  int ntowers;
  //  vector<HcalTrigTowerDetId> towerids;  
  double cell_energy;

  for(vector<DetId>::const_iterator itr = hbCells.begin(); itr != hbCells.end(); ++itr)
    {
      calib = 117;
      time = 8.4;
      HcalDetId barrelId(*itr);
      towerids = theTrigTowerGeometry.towerIds(barrelId);
      ncells = Tower_map.count(towerids[0]);
      ntowers = towerids.size();
      cell_energy = (energy*ntowers)/(ncells*calib);
      PCaloHit barrelHit(barrelId.rawId(), cell_energy, time, 0);
      result->push_back(barrelHit);
    }
  for(vector<DetId>::const_iterator itr = heCells.begin(); itr != heCells.end(); ++itr)
    {
      calib = 178;
      time = 13.0;
      HcalDetId endcapId(*itr);
      towerids = theTrigTowerGeometry.towerIds(endcapId);
      ncells = Tower_map.count(towerids[0]);
      ntowers = towerids.size();
      cell_energy = (energy*ntowers)/(ncells*calib);
      PCaloHit endcapHit(endcapId.rawId(), cell_energy, time, 0);
      result->push_back(endcapHit);
    }
  for(vector<DetId>::const_iterator itr = hfCells.begin(); itr != hfCells.end(); ++itr)
    {
      calib = 2.84;
      time = 37.0;
      HcalDetId forwardId(*itr);
      towerids = theTrigTowerGeometry.towerIds(forwardId);
      ncells = Tower_map.count(towerids[0]);
      ntowers = towerids.size();
      cell_energy = (energy*ntowers)/(ncells*calib);
      PCaloHit forwardHit(forwardId.rawId(), cell_energy, time, 0);
      result->push_back(forwardHit);
    }
  e.put(result,"HcalHits");
  energy += step_size;
}





