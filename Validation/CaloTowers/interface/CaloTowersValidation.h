#ifndef CaloTowersValidation_H
#define CaloTowersValidation_H
 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>
#include "DQMServices/Core/interface/MonitorElement.h"


class CaloTowersValidation : public edm::EDAnalyzer {
 public:
   CaloTowersValidation(edm::ParameterSet const& conf);
  ~CaloTowersValidation();
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void beginJob() ;
  virtual void endJob() ;

 private:
  double dR(double eta1, double phi1, double eta2, double phi2);
   
  DQMStore* dbe_;
  std::string outputFile_;
  std::string hcalselector_;

  typedef math::RhoEtaPhiVector Vector;

  std::string theCaloTowerCollectionLabel;

  int isub;
  int nevent;
  // eta limits to calcualte MET, SET (not to include HF if not needed)
  double etaMax[3];
  double etaMin[3];

  // ieta scan
  MonitorElement*  emean_vs_ieta_E;
  MonitorElement*  emean_vs_ieta_H;
  MonitorElement*  emean_vs_ieta_EH;

  MonitorElement*  emean_vs_ieta_E1;
  MonitorElement*  emean_vs_ieta_H1;
  MonitorElement*  emean_vs_ieta_EH1;

  MonitorElement* Ntowers_vs_ieta;
  MonitorElement* occupancy_map;
  MonitorElement* occupancy_vs_ieta;

  // Global maps
  MonitorElement*  mapEnergy_E;
  MonitorElement*  mapEnergy_H;
  MonitorElement*  mapEnergy_EH;
  MonitorElement*  mapEnergy_N;

  // HB
  MonitorElement* meEnergyHcalvsEcal_HB;
  MonitorElement* meEnergyHO_HB; 
  MonitorElement* meEnergyEcal_HB; 
  MonitorElement* meEnergyHcal_HB; 
  MonitorElement* meNumFiredTowers_HB;

  MonitorElement* meEnergyEcalTower_HB;
  MonitorElement* meEnergyHcalTower_HB;
  MonitorElement* meTotEnergy_HB;

  MonitorElement* mapEnergy_HB;
  MonitorElement* mapEnergyEcal_HB;
  MonitorElement* mapEnergyHcal_HB;
  MonitorElement* MET_HB;
  MonitorElement* SET_HB;
  MonitorElement* phiMET_HB;

  // HE
  MonitorElement* meEnergyHcalvsEcal_HE;
  MonitorElement* meEnergyHO_HE; 
  MonitorElement* meEnergyEcal_HE; 
  MonitorElement* meEnergyHcal_HE; 
  MonitorElement* meNumFiredTowers_HE;

  MonitorElement* meEnergyEcalTower_HE;
  MonitorElement* meEnergyHcalTower_HE;
  MonitorElement* meTotEnergy_HE;

  MonitorElement* mapEnergy_HE;
  MonitorElement* mapEnergyEcal_HE;
  MonitorElement* mapEnergyHcal_HE;
  MonitorElement* MET_HE;
  MonitorElement* SET_HE;
  MonitorElement* phiMET_HE;

  // HF
  MonitorElement* meEnergyHcalvsEcal_HF;
  MonitorElement* meEnergyHO_HF; 
  MonitorElement* meEnergyEcal_HF; 
  MonitorElement* meEnergyHcal_HF; 
  MonitorElement* meNumFiredTowers_HF;

  MonitorElement* meEnergyEcalTower_HF;
  MonitorElement* meEnergyHcalTower_HF;
  MonitorElement* meTotEnergy_HF;

  MonitorElement* mapEnergy_HF;
  MonitorElement* mapEnergyEcal_HF;
  MonitorElement* mapEnergyHcal_HF;
  MonitorElement* MET_HF;
  MonitorElement* SET_HF;
  MonitorElement* phiMET_HF;


};

#endif
