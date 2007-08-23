#ifndef ECALRECHITANALYZER_H
#define ECALRECHITANALYZER_H

// author: Bobby Scurlock (The University of Florida)
// date: 11/20/2006

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
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
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

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

//--egamma Reco stuff--//
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

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

class ECALRecHitAnalyzer : public edm::EDAnalyzer {
public:

  ECALRecHitAnalyzer(const edm::ParameterSet&);
  //~ECALRecHitAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

  void WriteECALRecHits(const edm::Event&, const edm::EventSetup&);
  void FillGeometry(const edm::EventSetup&);
  void DumpGeometry();
  void BookHistos();
 
 private:
  bool debug_;
   //Histo Files 
  TFile *m_DataFile, *m_GeomFile;
  // Geometry Histograms
  //--ECAL Barrel
  TH2F *hEB_ieta_iphi_etaMap;
  TH2F *hEB_ieta_iphi_phiMap;
  TH1F *hEB_ieta_detaMap;
  TH1F *hEB_ieta_dphiMap;
  //--ECAL +endcaps
  TH2F *hEEpZ_ix_iy_xMap;
  TH2F *hEEpZ_ix_iy_yMap;
  TH2F *hEEpZ_ix_iy_dxMap;
  TH2F *hEEpZ_ix_iy_dyMap;
  //--ECAL -endcaps
  TH2F *hEEmZ_ix_iy_xMap;
  TH2F *hEEmZ_ix_iy_yMap;
  TH2F *hEEmZ_ix_iy_dxMap;
  TH2F *hEEmZ_ix_iy_dyMap;

  // Data Histograms
  TH2F *hEEpZ_energy_ix_iy;
  TH2F *hEEmZ_energy_ix_iy;
  TH2F *hEB_energy_ieta_iphi;
  TH2F *hEEpZ_Occ_ix_iy;
  TH2F *hEEmZ_Occ_ix_iy;
  TH2F *hEB_Occ_ieta_iphi;

  int CurrentEvent;
};

#endif
