#ifndef HGCalDigiValidation_h
#define HGCalDigiValidation_h
// -*- C++ -*-
//
// Package:    HGCalDigiValidation
// Class:      HGCalDigiValidation
// 
/**\class HGCalDigiValidation HGCalDigiValidation.cc Validation/HGCalValidation/plugins/HGCalDigiValidation.cc

 Description: Validates SimHits of High Granularity Calorimeter

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Raman Khurana
//         Created:  Fri, 31 Jan 2014 18:35:18 GMT
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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "Geometry/FCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include <CLHEP/Geometry/Transform3D.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class HGCalDigiValidation : public edm::EDAnalyzer {

public:
  struct digiInfo{
    digiInfo() {
      x = y = z = 0.0;
      layer = adc = charge = 0;
    }
    double x, y, z;
    int layer, charge, adc;
  };

  explicit HGCalDigiValidation(const edm::ParameterSet&);
  ~HGCalDigiValidation();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void FillDigiInfo(digiInfo&   hinfo);
  void FillDigiInfo();
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  template<class T1, class T2> 
  void HGCDigiValidation(T1 detId, const HGCalGeometry& geom0, const T2 it);

private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------
  std::string       nameDetector_, DigiSource_;
  int               verbosity_, SampleIndx_;
  DQMStore          *dbe_;
  HGCalDDDConstants *hgcons_;
  int               layers_;

  std::map<int, int> OccupancyMap_plus_;
  std::map<int, int> OccupancyMap_minus_;

  std::vector<MonitorElement*> charge_;
  std::vector<MonitorElement*> DigiOccupancy_XY_;
  std::vector<MonitorElement*> ADC_;
  std::vector<MonitorElement*> DigiOccupancy_Plus_;
  std::vector<MonitorElement*> DigiOccupancy_Minus_;
  MonitorElement* MeanDigiOccupancy_Plus_;
  MonitorElement* MeanDigiOccupancy_Minus_;

};
#endif
