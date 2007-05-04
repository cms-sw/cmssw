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

using namespace std;

HcalDummyHitProducer::HcalDummyHitProducer(const edm::ParameterSet& ps)
{
  energyEM = 0;
  produces<edm::PCaloHitContainer>("HcalHits");
  step_size = ps.getParameter<double>("step_size");
  energyHad = ps.getParameter<double>("energy_start");
}

HcalDummyHitProducer::~HcalDummyHitProducer() {}

float sample_factor(HcalDetId id)
{
  float output;
  if (id.subdet() == 1)
    {
      if (id.ietaAbs() == 1) {output = 120.49;}
      else if (id.ietaAbs() == 2) {output = 120.49;}
      else if (id.ietaAbs() == 3) {output = 120.49;}
      else if (id.ietaAbs() == 4) {output = 120.58;}
      else if (id.ietaAbs() == 5) {output = 120.67;}
      else if (id.ietaAbs() == 6) {output = 120.76;}
      else if (id.ietaAbs() == 7) {output = 120.87;}
      else if (id.ietaAbs() == 8) {output = 120.98;}
      else if (id.ietaAbs() == 9) {output = 121.08;}
      else if (id.ietaAbs() == 10) {output = 121.44;}
      else if (id.ietaAbs() == 11) {output = 121.81;}
      else if (id.ietaAbs() == 12) {output = 122.17;}
      else if (id.ietaAbs() == 13) {output = 123.37;}
      else if (id.ietaAbs() == 14) {output = 124.57;}
      else if (id.ietaAbs() == 15) {output = 125.76;}
      else {output = 130.0;}
    }
  else if (id.subdet() == 2)
    {
      if (id.ietaAbs() == 15) {output = 151.05;}
      else if (id.ietaAbs() == 16) {output = 189.25;}
      else if (id.ietaAbs() == 17) {output = 174.51;}
      else if (id.ietaAbs() == 18) {output = 174.51;}
      else if (id.ietaAbs() == 19) {output = 174.51;}
      else if (id.ietaAbs() == 20) {output = 174.51;}
      else if (id.ietaAbs() == 21) {output = 174.51;}
      else if (id.ietaAbs() == 22) {output = 174.51;}
      else if (id.ietaAbs() == 23) {output = 174.51;}
      else if (id.ietaAbs() == 24) {output = 174.51;}
      else if (id.ietaAbs() == 25) {output = 174.51;}
      else if (id.ietaAbs() == 26) {output = 174.51;}
      else if (id.ietaAbs() == 27) {output = 174.51;}
      else {output = 174.51;}
    }
  else if (id.subdet() == 4)
    {
      if(id.depth() == 1) {output = 2.79;}
      else {output =  1.842;}
    }
  else
    {
      output =  0.0;
    }
  return output;
}

void HcalDummyHitProducer::produce(edm::Event& e, const edm::EventSetup& eventSetup) {
  // get the appropriate gains, noises, & widths for this event
  //edm::ESHandle<HcalDbService> conditions;
  //eventSetup.get<HcalDbRecord>().get(conditions);
  //theParameters->setDbService(conditions.product());

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
  double cell_energyEM, cell_energyHad;

  for(vector<DetId>::const_iterator itr = hbCells.begin(); itr != hbCells.end(); ++itr)
    {
      HcalDetId barrelId(*itr);
      //const CaloSimParameters & parameters = theParameterMap->simParameters(*itr);
      //calib = parameters.simHitToPhotoelectrons()
      //    * parameters.photoelectronsToAnalog()
      //    * theParameters->fCtoGeV(*itr);
      //time = 8.4 - parameters.timePhase(); 
      calib = sample_factor(barrelId);
      time = 8.4;
      towerids = theTrigTowerGeometry.towerIds(barrelId);
      ncells = Tower_map.count(towerids[0]);
      ntowers = towerids.size();
      cell_energyEM = 0;
      //cell_energyHad= (energyHad*ntowers)/(ncells*calib);
      if(barrelId.depth()==1)
	{
	  cell_energyHad= (energyHad*ntowers)/(calib);
	}
      else
	{
	  cell_energyHad = 0;
	}
      PCaloHit barrelHit(barrelId.rawId(), cell_energyEM, cell_energyHad, time, 0);
      result->push_back(barrelHit);
    }
  for(vector<DetId>::const_iterator itr = heCells.begin(); itr != heCells.end(); ++itr)
    {
      HcalDetId endcapId(*itr);
      //const CaloSimParameters & parameters = theParameterMap->simParameters(*itr);
      //calib = parameters.simHitToPhotoelectrons()
      //* parameters.photoelectronsToAnalog()
      //* theParameters->fCtoGeV(*itr);
      //time = 13.0 - parameters.timePhase();
      calib = sample_factor(endcapId);
      time = 13.0;
      towerids = theTrigTowerGeometry.towerIds(endcapId);
      ncells = Tower_map.count(towerids[0]);
      ntowers = towerids.size();
      cell_energyEM =  0;
      //cell_energyHad= (energyHad*ntowers)/(ncells*calib);
      if(endcapId.depth()==1)
        {
          cell_energyHad= (energyHad*ntowers)/(calib);
        }
      else
        {
          cell_energyHad = 0;
        }
      PCaloHit endcapHit(endcapId.rawId(), cell_energyEM, cell_energyHad, time, 0);
      result->push_back(endcapHit);
    }
  for(vector<DetId>::const_iterator itr = hfCells.begin(); itr != hfCells.end(); ++itr)
    {
      HcalDetId forwardId(*itr);
      //const CaloSimParameters & parameters = theParameterMap->simParameters(*itr);
      //calib = parameters.simHitToPhotoelectrons()
      //* parameters.photoelectronsToAnalog()
      //* theParameters->fCtoGeV(*itr);
      //time = 37.0 - parameters.timePhase();
      calib = sample_factor(forwardId);
      time = 37.0;
      towerids = theTrigTowerGeometry.towerIds(forwardId);
      ncells = Tower_map.count(towerids[0]);
      ntowers = towerids.size();
      cell_energyEM = 0;
      //cell_energyHad= (energyHad*ntowers)/(ncells*calib);
      if(forwardId.depth()==1)
        {
          cell_energyHad= (int)((energyHad)/(calib));
        }
      else
        {
          cell_energyHad = 0;
        }

      PCaloHit forwardHit(forwardId.rawId(), cell_energyEM, cell_energyHad, time, 0);
      result->push_back(forwardHit);
    }
  e.put(result,"HcalHits");
  energyHad += step_size;
}





