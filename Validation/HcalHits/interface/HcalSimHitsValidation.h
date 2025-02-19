#ifndef HcalSimHitsValidation_H
#define HcalSimHitsValidation_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>
#include "DQMServices/Core/interface/MonitorElement.h"


class HcalSimHitsValidation : public edm::EDAnalyzer {

public:
  HcalSimHitsValidation(edm::ParameterSet const& conf);
  ~HcalSimHitsValidation();
  virtual void analyze(edm::Event const& ev, edm::EventSetup const& c);
  virtual void beginJob() ;
  virtual void endJob() ;

private:
  
  double dR(double eta1, double phi1, double eta2, double phi2);
  double phi12(double phi1, double en1, double phi2, double en2);
  double dPhiWsign(double phi1,double phi2);  

  DQMStore* dbe_;
  
  std::string outputFile_;

  // Hits counters
  MonitorElement* Nhb;
  MonitorElement* Nhe;
  MonitorElement* Nho;
  MonitorElement* Nhf;

  // In ALL other cases : 2D ieta-iphi maps 
  // without and with cuts (a la "Scheme B") on energy
  // - only in the cone around particle for single-part samples (mc = "yes")
  // - for all calls in milti-particle samples (mc = "no")

  MonitorElement* emean_vs_ieta_HB1;
  MonitorElement* emean_vs_ieta_HB2;
  MonitorElement* emean_vs_ieta_HE1;
  MonitorElement* emean_vs_ieta_HE2;
  MonitorElement* emean_vs_ieta_HE3;
  MonitorElement* emean_vs_ieta_HO;
  MonitorElement* emean_vs_ieta_HF1;
  MonitorElement* emean_vs_ieta_HF2;

  MonitorElement* occupancy_vs_ieta_HB1;
  MonitorElement* occupancy_vs_ieta_HB2;
  MonitorElement* occupancy_vs_ieta_HE1;
  MonitorElement* occupancy_vs_ieta_HE2;
  MonitorElement* occupancy_vs_ieta_HE3;
  MonitorElement* occupancy_vs_ieta_HO;
  MonitorElement* occupancy_vs_ieta_HF1;
  MonitorElement* occupancy_vs_ieta_HF2;

  // for single monoenergetic particles - cone collection profile vs ieta.
  MonitorElement* meEnConeEtaProfile;
  MonitorElement* meEnConeEtaProfile_E;
  MonitorElement* meEnConeEtaProfile_EH;

  // energy of rechits
  MonitorElement* meSimHitsEnergyHB;
  MonitorElement* meSimHitsEnergyHE;
  MonitorElement* meSimHitsEnergyHO;
  MonitorElement* meSimHitsEnergyHF;

  edm::ESHandle<CaloGeometry> geometry ;

  // counter
  int nevtot;

};

#endif
