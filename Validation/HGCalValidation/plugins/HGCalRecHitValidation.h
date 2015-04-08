#ifndef HGCalRecHitValidation_h
#define HGCalRecHitValidation_h
// -*- C++ -*-
//
// Package:    HGCalRecHitValidation
// Class:      HGCalRecHitValidation
// 
/**\class HGCalRecHitValidation HGCalRecHitValidation.cc Validation/HGCalValidation/plugins/HGCalRecHitValidation.cc

 Description: Validates SimHits of High Granularity Calorimeter

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Raman Khurana
//         Created:  Sunday, 17th Augst 2014 11:30:15 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class HGCalRecHitValidation : public edm::EDAnalyzer {

public:
  struct energysum{
    energysum() {e15=e25=e50=e100=e250=e1000=0.0;}
    double e15, e25, e50, e100, e250, e1000;
  };

  struct HitsInfo{
    HitsInfo() {
      x=y=z=time=energy=phi=eta=0.0;
      //cell=sector=
      layer=0;
    }
    float x, y, z, time, energy, phi, eta ;
    float layer;
    //    int    cell, sector, layer;
  };
  

  explicit HGCalRecHitValidation(const edm::ParameterSet&);
  ~HGCalRecHitValidation();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  template<class T1, class T2>
    void recHitValidation(DetId & detId, int layer, const T1* geom, T2 it);
  void fillHitsInfo(); 
  void fillHitsInfo(HitsInfo& hits); 
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  
private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------
  std::string           nameDetector_;
  edm::InputTag         recHitSource_;
  DQMStore              *dbe_;
  int                   verbosity_;
  unsigned int          layers_;
  std::map<int, int>    OccupancyMap_plus;
  std::map<int, int>    OccupancyMap_minus;

  std::vector<MonitorElement*> EtaPhi_Plus_;
  std::vector<MonitorElement*> EtaPhi_Minus_;
  std::vector<MonitorElement*> energy_;
  std::vector<MonitorElement*> HitOccupancy_Plus_;
  std::vector<MonitorElement*> HitOccupancy_Minus_;
  MonitorElement* MeanHitOccupancy_Plus_;
  MonitorElement* MeanHitOccupancy_Minus_;
};
#endif
