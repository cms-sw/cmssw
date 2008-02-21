#ifndef HcalRecHitsValidation_H
#define HcalRecHitsValidation_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "FWCore/Framework/interface/Selector.h"

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>


class HcalRecHitsValidation : public edm::EDAnalyzer {
 public:
  HcalRecHitsValidation(edm::ParameterSet const& conf);
  ~HcalRecHitsValidation();
  virtual void analyze(edm::Event const& ev, edm::EventSetup const& c);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;
 private:
  
  virtual void fillRecHitsTmp(int subdet_, edm::Event const& ev);
  double dR(double eta1, double phi1, double eta2, double phi2);
  double phi12(double phi1, double en1, double phi2, double en2);
  double dPhiWsign(double phi1,double phi2);  

  DaqMonitorBEInterface* dbe_;
  
  std::string outputFile_;
  std::string hcalselector_;
  std::string ecalselector_;
  std::string eventype_;
  std::string sign_;
  std::string mc_;

  // choice of subdetector in config : noise/HB/HE/HO/HF/ALL (0/1/2/3/4/5)
  int subdet_;

  // single/multi-particle sample (1/2)
  int etype_;
  int iz;
  int imc;

  // ZS-specific

  MonitorElement* ZS_HB1;
  MonitorElement* ZS_HB2;
  MonitorElement* ZS_HE1;
  MonitorElement* ZS_HE2;
  MonitorElement* ZS_HE3;
  MonitorElement* ZS_HO;
  MonitorElement* ZS_HF1;
  MonitorElement* ZS_HF2;

  MonitorElement* ZS_nHB1;
  MonitorElement* ZS_nHB2;
  MonitorElement* ZS_nHE1;
  MonitorElement* ZS_nHE2;
  MonitorElement* ZS_nHE3;
  MonitorElement* ZS_nHO;
  MonitorElement* ZS_nHF1;
  MonitorElement* ZS_nHF2;

  MonitorElement* ZS_seqHB1;
  MonitorElement* ZS_seqHB2;
  MonitorElement* ZS_seqHE1;
  MonitorElement* ZS_seqHE2;
  MonitorElement* ZS_seqHE3;
  MonitorElement* ZS_seqHO;
  MonitorElement* ZS_seqHF1;
  MonitorElement* ZS_seqHF2;

  // In ALL other cases : 2D ieta-iphi maps 
  // without and with cuts (a la "Scheme B") on energy
  MonitorElement* map_ecal;

  MonitorElement* map_depth1;
  MonitorElement* map_depth2;
  MonitorElement* map_depth3;
  MonitorElement* map_depth4;

  MonitorElement* map_depth1_cuts;
  MonitorElement* map_depth2_cuts;
  MonitorElement* map_depth3_cuts;
  MonitorElement* map_depth4_cuts;

  MonitorElement* profile_z1;
  MonitorElement* profile_z2;
  MonitorElement* profile_z3;
  MonitorElement* profile_z4;

  MonitorElement* meEnConeEtaProfile_depth1;
  MonitorElement* meEnConeEtaProfile_depth2;
  MonitorElement* meEnConeEtaProfile_depth3;
  MonitorElement* meEnConeEtaProfile_depth4;

  // also - energy in the cone around MC particle
  MonitorElement* map_econe_depth1;
  MonitorElement* map_econe_depth2;
  MonitorElement* map_econe_depth3;
  MonitorElement* map_econe_depth4;
 
  // Single particles - deviation of cluster from MC truth
  MonitorElement* meDeltaPhi;
  MonitorElement* meDeltaEta;
  MonitorElement* meDeltaPhiS;  // simcluster
  MonitorElement* meDeltaEtaS;  // simculster

  //----------- NOISE case
  MonitorElement* e_hb;
  MonitorElement* e_he;
  MonitorElement* e_ho;
  MonitorElement* e_hfl;
  MonitorElement* e_hfs;

  // number of rechits above threshold 1GEV
  MonitorElement* meNumRecHitsThreshHB;
  MonitorElement* meNumRecHitsThreshHE;
  MonitorElement* meNumRecHitsThreshHO;

  // number of rechits in the cone
  MonitorElement* meNumRecHitsConeHB;
  MonitorElement* meNumRecHitsConeHE;
  MonitorElement* meNumRecHitsConeHO;
  MonitorElement* meNumRecHitsConeHF;

  // time?
  MonitorElement* meTimeHB;
  MonitorElement* meTimeHE;
  MonitorElement* meTimeHO;
  MonitorElement* meTimeHF;

  // energy of rechits
  MonitorElement* meRecHitsEnergyHB;
  MonitorElement* meRecHitsEnergyHE;
  MonitorElement* meRecHitsEnergyHO;
  MonitorElement* meRecHitsEnergyHF;

  MonitorElement* meTE_HB;
  MonitorElement* meTE_HB1;
  MonitorElement* meTE_HB2;
  MonitorElement* meTEprofileHB;
  MonitorElement* meTE_HE;
  MonitorElement* meTE_HE1;
  MonitorElement* meTE_HE2;
  MonitorElement* meTEprofileHE;
  MonitorElement* meTE_HO;
  MonitorElement* meTEprofileHO;
  MonitorElement* meTE_HF;
  MonitorElement* meTE_HFL;
  MonitorElement* meTE_HFS;
  MonitorElement* meTEprofileHF;


  MonitorElement* meSumRecHitsEnergyHB;
  MonitorElement* meSumRecHitsEnergyHE;
  MonitorElement* meSumRecHitsEnergyHO;
  MonitorElement* meSumRecHitsEnergyHF;


  MonitorElement* meSumRecHitsEnergyConeHB;
  MonitorElement* meSumRecHitsEnergyConeHE;
  MonitorElement* meSumRecHitsEnergyConeHO;
  MonitorElement* meSumRecHitsEnergyConeHF;
  MonitorElement* meSumRecHitsEnergyConeHFL;
  MonitorElement* meSumRecHitsEnergyConeHFS;


  MonitorElement* meEcalHcalEnergyHB;
  MonitorElement* meEcalHcalEnergyHE;
   
  MonitorElement* meEcalHcalEnergyConeHB; 
  MonitorElement* meEcalHcalEnergyConeHE; 
  MonitorElement* meEcalHcalEnergyConeHO; 
  MonitorElement* meEcalHcalEnergyConeHF; 
 
  // Histo (2D plot) for sum of RecHits vs SimHits (hcal only)
  MonitorElement* meRecHitSimHitHB;
  MonitorElement* meRecHitSimHitHE;
  MonitorElement* meRecHitSimHitHO;
  MonitorElement* meRecHitSimHitHF;
  MonitorElement* meRecHitSimHitHFL;
  MonitorElement* meRecHitSimHitHFS;
  // profile histo (2D plot) for sum of RecHits vs SimHits (hcal only)
  MonitorElement* meRecHitSimHitProfileHB;
  MonitorElement* meRecHitSimHitProfileHE;
  MonitorElement* meRecHitSimHitProfileHO;
  MonitorElement* meRecHitSimHitProfileHF;
  MonitorElement* meRecHitSimHitProfileHFL;
  MonitorElement* meRecHitSimHitProfileHFS;

  // 2D plot of sum of RecHits in HCAL as function of ECAL's one
  MonitorElement* meEnergyHcalVsEcalHB;
  MonitorElement* meEnergyHcalVsEcalHE;
  
  // number of ECAL's rechits in cone 0.3 
  MonitorElement* meNumEcalRecHitsConeHB;
  MonitorElement* meNumEcalRecHitsConeHE;

  edm::ESHandle<CaloGeometry> geometry ;

 // Filling vectors with essential RecHits data
  std::vector<int>    csub;
  std::vector<int>    cieta;
  std::vector<int>    ciphi;
  std::vector<int>    cdepth;
  std::vector<double> cen;
  std::vector<double> ceta;
  std::vector<double> cphi;
  std::vector<double> ctime;
  std::vector<double> cz;

  // array or min. e-values  ieta x iphi x depth x subdet
  double emap_min[82][72][4][4];

};

#endif
