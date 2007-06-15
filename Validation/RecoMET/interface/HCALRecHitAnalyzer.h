#ifndef HCALRECHITANALYZER_H
#define HCALRECHITANALYZER_H

// author: Bobby Scurlock (The University of Florida)
// date: 8/24/2006

#include <memory>
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"


#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TMath.h>

class DetId;
//class HcalTopology;
class CaloGeometry;
class CaloSubdetectorGeometry;
//class CaloTowerConstituentsMap;
//class CaloRecHit;


//
// class declaration
//

class HCALRecHitAnalyzer : public edm::EDAnalyzer {
public:

  HCALRecHitAnalyzer(const edm::ParameterSet&);
  //~HCALRecHitAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

  void WriteHCALRecHits(const edm::Event&, const edm::EventSetup&);
  void FillGeometry(const edm::EventSetup&);
  void DumpGeometry();
  void BookHistos();
 
 private:
  bool debug_;
  string outputFile_;

  //Histo File 
  TFile *m_DataFile, *m_GeomFile;
  // Geometry Histograms
  TH2F *hCT_ieta_iphi_etaMap;
  TH2F *hCT_ieta_iphi_phiMap;
  TH1F *hCT_ieta_detaMap;
  TH1F *hCT_ieta_dphiMap;
  // Data Histograms
  TH2F *hHCAL_L1_energy_ieta_iphi;
  TH2F *hHCAL_L2_energy_ieta_iphi;
  TH2F *hHCAL_L3_energy_ieta_iphi;
  TH2F *hHCAL_L4_energy_ieta_iphi;
  TH2F *hHCAL_L1_Occ_ieta_iphi;
  TH2F *hHCAL_L2_Occ_ieta_iphi;
  TH2F *hHCAL_L3_Occ_ieta_iphi;
  TH2F *hHCAL_L4_Occ_ieta_iphi;
  
  int CurrentEvent;
  float EnergyThreshold;
  int theEvent;
  int FirstEvent;
  int LastEvent;
};

#endif
