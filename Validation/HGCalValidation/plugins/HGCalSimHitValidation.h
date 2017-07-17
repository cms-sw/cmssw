#ifndef HGCalSimHitValidation_h
#define HGCalSimHitValidation_h
// -*- C++ -*-
//
// Package:    HGCalSimHitValidation
// Class:      HGCalSimHitValidation
// 
/**\class HGCalSimHitValidation HGCalSimHitValidation.cc Validation/HGCalValidation/plugins/HGCalSimHitValidation.cc

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
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"
#include <CLHEP/Geometry/Transform3D.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>

class HGCalSimHitValidation : public DQMEDAnalyzer {
  
public:
  
  struct energysum{
    energysum() {etotal=0; for (int i=0; i<6; ++i) eTime[i] = 0.;}
    double eTime[6], etotal;
  };
  
  struct hitsinfo{
    hitsinfo() {
      x=y=z=phi=eta=0.0;
      cell=sector=layer=0;
    }
    double x, y, z, phi, eta;
    int    cell, sector, layer;
  };
  
  
  explicit HGCalSimHitValidation(const edm::ParameterSet&);
  ~HGCalSimHitValidation();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  
private:

  void analyzeHits (std::vector<PCaloHit>& hits);
  void fillOccupancyMap(std::map<int, int>& OccupancyMap, int layer);
  void fillHitsInfo(std::pair<hitsinfo,energysum> hit_, unsigned int itimeslice, double esum); 
  bool defineGeometry(edm::ESTransientHandle<DDCompactView> &ddViewH);
  
  // ----------member data ---------------------------
  std::string                nameDetector_, caloHitSource_;
  const HGCalDDDConstants   *hgcons_;
  const HcalDDDRecConstants *hcons_;
  int                        verbosity_;
  bool                       heRebuild_, testNumber_, symmDet_;
  std::vector<double>        times_;
  unsigned int               nTimes_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hits_;
  edm::EDGetTokenT<edm::HepMCProduct>      tok_hepMC_;
  unsigned int              layers_;
  std::map<uint32_t, HepGeom::Transform3D> transMap_;
  
  std::vector<MonitorElement*> HitOccupancy_Plus_, HitOccupancy_Minus_;
  std::vector<MonitorElement*> EtaPhi_Plus_,  EtaPhi_Minus_;
  MonitorElement              *MeanHitOccupancy_Plus_, *MeanHitOccupancy_Minus_;
  std::vector<MonitorElement*> energy_[6];
};
#endif
