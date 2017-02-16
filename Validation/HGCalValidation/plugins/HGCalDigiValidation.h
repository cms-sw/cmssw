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
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class HGCalDigiValidation : public DQMEDAnalyzer {

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
  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void fillDigiInfo(digiInfo&   hinfo);
  void fillDigiInfo();
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  template<class T1, class T2> 
  void digiValidation(const T1& detId, const T2* geom, int, uint16_t, double);
  
  // ----------member data ---------------------------
  std::string       nameDetector_;
  edm::EDGetToken   digiSource_;
  int               verbosity_, SampleIndx_;
  bool              ifHCAL_;
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
