#ifndef HcalRecHitsValidation_H
#define HcalRecHitsValidation_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
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

#include "DataFormats/DetId/interface/DetId.h"
// channel status
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"

// severity level assignment for HCAL
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

// severity level assignment for ECAL
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"


class HcalRecHitsValidation : public DQMEDAnalyzer {
 public:
  HcalRecHitsValidation(edm::ParameterSet const& conf);
  ~HcalRecHitsValidation();
  virtual void analyze(edm::Event const& ev, edm::EventSetup const& c);

  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);

 private:
  
  virtual void fillRecHitsTmp(int subdet_, edm::Event const& ev);
  double dR(double eta1, double phi1, double eta2, double phi2);
  double phi12(double phi1, double en1, double phi2, double en2);
  double dPhiWsign(double phi1,double phi2);  

  std::string outputFile_;
  std::string hcalselector_;
  std::string ecalselector_;
  std::string eventype_;
  std::string sign_;
  std::string mc_;
  bool        famos_;
  bool        useAllHistos_;

  //RecHit Collection input tags
  edm::EDGetTokenT<edm::HepMCProduct> tok_evt_;
  edm::EDGetTokenT<EBRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EERecHitCollection> tok_EE_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hh_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;

  // choice of subdetector in config : noise/HB/HE/HO/HF/ALL (0/1/2/3/4/5)
  int subdet_;

  // single/multi-particle sample (1/2)
  int etype_;
  int iz;
  int imc;

  // for checking the status of ECAL and HCAL channels stored in the DB 
  const HcalChannelQuality* theHcalChStatus;
  // calculator of severety level for HCAL
  const HcalSeverityLevelComputer* theHcalSevLvlComputer;
  int hcalSevLvl(const CaloRecHit* hit);

  std::vector<int> hcalHBSevLvlVec, hcalHESevLvlVec, hcalHFSevLvlVec, hcalHOSevLvlVec;

  MonitorElement* sevLvl_HB;
  MonitorElement* sevLvl_HE;
  MonitorElement* sevLvl_HF;
  MonitorElement* sevLvl_HO; 

  // RecHits counters
  MonitorElement* Nhb;
  MonitorElement* Nhe;
  MonitorElement* Nho;
  MonitorElement* Nhf;

  // ZS-specific

  MonitorElement* map_depth1;
  MonitorElement* map_depth2;
  MonitorElement* map_depth3;
  MonitorElement* map_depth4;

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
  // - only in the cone around particle for single-part samples (mc = "yes")
  // - for all calls in milti-particle samples (mc = "no")

  MonitorElement* map_ecal;

  MonitorElement* emap_depth1;
  MonitorElement* emap_depth2;
  MonitorElement* emap_depth3;
  MonitorElement* emap_depth4;

  MonitorElement* emean_vs_ieta_HB1;
  MonitorElement* emean_vs_ieta_HB2;
  MonitorElement* emean_vs_ieta_HE1;
  MonitorElement* emean_vs_ieta_HE2;
  MonitorElement* emean_vs_ieta_HE3;
  MonitorElement* emean_vs_ieta_HO;
  MonitorElement* emean_vs_ieta_HF1;
  MonitorElement* emean_vs_ieta_HF2;

  MonitorElement* RMS_vs_ieta_HB1;
  MonitorElement* RMS_vs_ieta_HB2;
  MonitorElement* RMS_vs_ieta_HE1;
  MonitorElement* RMS_vs_ieta_HE2;
  MonitorElement* RMS_vs_ieta_HE3;
  MonitorElement* RMS_vs_ieta_HO;
  MonitorElement* RMS_vs_ieta_HF1;
  MonitorElement* RMS_vs_ieta_HF2;

  MonitorElement* emean_seqHB1;
  MonitorElement* emean_seqHB2;
  MonitorElement* emean_seqHE1;
  MonitorElement* emean_seqHE2;
  MonitorElement* emean_seqHE3;
  MonitorElement* emean_seqHO;
  MonitorElement* emean_seqHF1;
  MonitorElement* emean_seqHF2;

  MonitorElement* RMS_seq_HB1;
  MonitorElement* RMS_seq_HB2;
  MonitorElement* RMS_seq_HE1;
  MonitorElement* RMS_seq_HE2;
  MonitorElement* RMS_seq_HE3;
  MonitorElement* RMS_seq_HO;
  MonitorElement* RMS_seq_HF1;
  MonitorElement* RMS_seq_HF2;

  MonitorElement* occupancy_map_HB1;
  MonitorElement* occupancy_map_HB2;
  MonitorElement* occupancy_map_HE1;
  MonitorElement* occupancy_map_HE2;
  MonitorElement* occupancy_map_HE3;
  MonitorElement* occupancy_map_HO;
  MonitorElement* occupancy_map_HF1;
  MonitorElement* occupancy_map_HF2;

  MonitorElement* occupancy_vs_ieta_HB1;
  MonitorElement* occupancy_vs_ieta_HB2;
  MonitorElement* occupancy_vs_ieta_HE1;
  MonitorElement* occupancy_vs_ieta_HE2;
  MonitorElement* occupancy_vs_ieta_HE3;
  MonitorElement* occupancy_vs_ieta_HO;
  MonitorElement* occupancy_vs_ieta_HF1;
  MonitorElement* occupancy_vs_ieta_HF2;

  MonitorElement* occupancy_seqHB1;
  MonitorElement* occupancy_seqHB2;
  MonitorElement* occupancy_seqHE1;
  MonitorElement* occupancy_seqHE2;
  MonitorElement* occupancy_seqHE3;
  MonitorElement* occupancy_seqHO;
  MonitorElement* occupancy_seqHF1;
  MonitorElement* occupancy_seqHF2;


  // also - energy in the cone around MC particle
  MonitorElement* map_econe_depth1;
  MonitorElement* map_econe_depth2;
  MonitorElement* map_econe_depth3;
  MonitorElement* map_econe_depth4;
 
  // for single monoenergetic particles - cone collection profile vs ieta.
  MonitorElement* meEnConeEtaProfile_depth1;
  MonitorElement* meEnConeEtaProfile_depth2;
  MonitorElement* meEnConeEtaProfile_depth3;
  MonitorElement* meEnConeEtaProfile_depth4;
  MonitorElement* meEnConeEtaProfile;
  MonitorElement* meEnConeEtaProfile_E;
  MonitorElement* meEnConeEtaProfile_EH;
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

  MonitorElement* meTE_Low_HB;
  MonitorElement* meTE_HB;
  MonitorElement* meTE_High_HB;
  MonitorElement* meTE_HB1;
  MonitorElement* meTE_HB2;
  MonitorElement* meTEprofileHB_Low;
  MonitorElement* meTEprofileHB;
  MonitorElement* meTEprofileHB_High;

  MonitorElement* meTE_Low_HE;
  MonitorElement* meTE_HE;
  MonitorElement* meTE_HE1;
  MonitorElement* meTE_HE2;
  MonitorElement* meTEprofileHE_Low;
  MonitorElement* meTEprofileHE;

  MonitorElement* meTE_HO;
  MonitorElement* meTE_High_HO;
  MonitorElement* meTEprofileHO;
  MonitorElement* meTEprofileHO_High;

  MonitorElement* meTE_Low_HF;
  MonitorElement* meTE_HF;
  MonitorElement* meTE_HFL;
  MonitorElement* meTE_HFS;
  MonitorElement* meTEprofileHF_Low;
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

  //Status word histos
  MonitorElement* RecHit_StatusWord_HB;
  MonitorElement* RecHit_StatusWord_HE;
  MonitorElement* RecHit_StatusWord_HF;
  MonitorElement* RecHit_StatusWord_HF67;
  MonitorElement* RecHit_StatusWord_HO;

  //Status word correlation
  MonitorElement* RecHit_StatusWordCorr_HB;
  MonitorElement* RecHit_StatusWordCorr_HE;

  MonitorElement* RecHit_StatusWordCorrAll_HB;
  MonitorElement* RecHit_StatusWordCorrAll_HE;

  //Aux Status word histos
  MonitorElement* RecHit_Aux_StatusWord_HB;
  MonitorElement* RecHit_Aux_StatusWord_HE;
  MonitorElement* RecHit_Aux_StatusWord_HF;
  MonitorElement* RecHit_Aux_StatusWord_HO;

 // Filling vectors with essential RecHits data
  std::vector<int>      csub;
  std::vector<int>      cieta;
  std::vector<int>      ciphi;
  std::vector<int>      cdepth;
  std::vector<double>   cen;
  std::vector<double>   ceta;
  std::vector<double>   cphi;
  std::vector<double>   ctime;
  std::vector<double>   cz;
  std::vector<uint32_t> cstwd;
  std::vector<uint32_t> cauxstwd;

  // array or min. e-values  ieta x iphi x depth x subdet
  double emap_min[82][72][4][4];

  // counter
  int nevtot;

};

#endif
