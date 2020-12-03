#ifndef HcalRecHitsValidation_H
#define HcalRecHitsValidation_H

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/HcalCommonData/interface/HcalDDDRecConstants.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/HcalRecNumberingRecord.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include <algorithm>
#include <cmath>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/DetId/interface/DetId.h"
// channel status
#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "CondFormats/DataRecord/interface/HcalChannelQualityRcd.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"

// severity level assignment for HCAL
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"

// severity level assignment for ECAL
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

class HcalRecHitsValidation : public DQMEDAnalyzer {
public:
  HcalRecHitsValidation(edm::ParameterSet const &conf);
  ~HcalRecHitsValidation() override;
  void analyze(edm::Event const &ev, edm::EventSetup const &c) override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

private:
  virtual void fillRecHitsTmp(int subdet_, edm::Event const &ev);
  double dR(double eta1, double phi1, double eta2, double phi2);
  double phi12(double phi1, double en1, double phi2, double en2);
  double dPhiWsign(double phi1, double phi2);

  std::string topFolderName_;

  std::string outputFile_;
  std::string hcalselector_;
  std::string ecalselector_;
  std::string sign_;
  std::string mc_;
  bool testNumber_;

  // RecHit Collection input tags
  edm::EDGetTokenT<edm::HepMCProduct> tok_evt_;
  edm::EDGetTokenT<EBRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EERecHitCollection> tok_EE_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_hh_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;
  edm::EDGetTokenT<HFRecHitCollection> tok_hf_;
  edm::EDGetTokenT<HORecHitCollection> tok_ho_;

  edm::ESGetToken<HcalDDDRecConstants, HcalRecNumberingRecord> tok_HRNDC_;
  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_Geom_;

  // choice of subdetector in config : noise/HB/HE/HO/HF/ALL (0/1/2/3/4/5)
  int subdet_;

  // single/multi-particle sample (1/2)
  int iz;
  int imc;

  // In ALL other cases : 2D ieta-iphi maps
  // without and with cuts (a la "Scheme B") on energy
  // - only in the cone around particle for single-part samples (mc = "yes")
  // - for all calls in milti-particle samples (mc = "no")

  MonitorElement *meEnConeEtaProfile;
  MonitorElement *meEnConeEtaProfile_E;
  MonitorElement *meEnConeEtaProfile_EH;

  // energy of rechits
  MonitorElement *meRecHitsEnergyHB;
  MonitorElement *meRecHitsEnergyHE;
  MonitorElement *meRecHitsEnergyHO;
  MonitorElement *meRecHitsEnergyHF;

  MonitorElement *meTEprofileHB_Low;
  MonitorElement *meTEprofileHB;
  MonitorElement *meTEprofileHB_High;

  MonitorElement *meTEprofileHE_Low;
  MonitorElement *meTEprofileHE;

  MonitorElement *meTEprofileHO;
  MonitorElement *meTEprofileHO_High;

  MonitorElement *meTEprofileHF_Low;
  MonitorElement *meTEprofileHF;

  // Histo (2D plot) for sum of RecHits vs SimHits (hcal only)
  MonitorElement *meRecHitSimHitHB;
  MonitorElement *meRecHitSimHitHE;
  MonitorElement *meRecHitSimHitHO;
  MonitorElement *meRecHitSimHitHF;
  MonitorElement *meRecHitSimHitHFL;
  MonitorElement *meRecHitSimHitHFS;
  // profile histo (2D plot) for sum of RecHits vs SimHits (hcal only)
  MonitorElement *meRecHitSimHitProfileHB;
  MonitorElement *meRecHitSimHitProfileHE;
  MonitorElement *meRecHitSimHitProfileHO;
  MonitorElement *meRecHitSimHitProfileHF;
  MonitorElement *meRecHitSimHitProfileHFL;
  MonitorElement *meRecHitSimHitProfileHFS;

  // 2D plot of sum of RecHits in HCAL as function of ECAL's one
  MonitorElement *meEnergyHcalVsEcalHB;
  MonitorElement *meEnergyHcalVsEcalHE;

  // Chi2
  MonitorElement *meRecHitsM2Chi2HB;
  MonitorElement *meRecHitsM2Chi2HE;

  MonitorElement *meLog10Chi2profileHB;
  MonitorElement *meLog10Chi2profileHE;

  const CaloGeometry *geometry_;

  // Filling vectors with essential RecHits data
  std::vector<int> csub;
  std::vector<int> cieta;
  std::vector<int> ciphi;
  std::vector<int> cdepth;
  std::vector<double> cen;
  std::vector<double> ceta;
  std::vector<double> cphi;
  std::vector<double> ctime;
  std::vector<double> cz;
  std::vector<uint32_t> cstwd;
  std::vector<uint32_t> cauxstwd;
  std::vector<double> cchi2;

  // counter
  int nevtot;
};

#endif
