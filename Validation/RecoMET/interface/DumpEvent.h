#ifndef DUMPEVENT_H
#define DUMPEVENT_H

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

#include "DataFormats/MuonReco/interface/Muon.h"


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

class DumpEvent : public edm::EDAnalyzer {
public:

  DumpEvent(const edm::ParameterSet&);
  //~DumpEvent();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

  void WritePhotons(const edm::Event&, const edm::EventSetup&);
  void WriteElectrons(const edm::Event&, const edm::EventSetup&);
  void WriteJets(const edm::Event&, const edm::EventSetup&);
  void WriteMuons(const edm::Event&, const edm::EventSetup&);
  void WriteMET(const edm::Event&, const edm::EventSetup&);
  void WriteSCs(const edm::Event&, const edm::EventSetup&);
  void BookHistos();
 
 private:
  bool debug_;
   //Histo Files 
  TFile *m_DataFile;
  
  TH1F *hElectron_eta, *hElectron_phi, *hElectron_energy; 
  TH1F *hPhoton_eta, *hPhoton_phi, *hPhoton_energy; 
  TH1F *hJet_eta, *hJet_phi, *hJet_energy; 
  TH2F *hCaloTowerToJetMap_ieta_iphi;
  TH1F *hMuon_eta, *hMuon_phi, *hMuon_pt; 
  TH1F *hRecoMET_phi, *hRecoMET_MET;
  TH1F *hGenMET_phi;
  //----Superclusters
  TH2F *hEEpZ_SC_ix_iy;
  TH2F *hEEmZ_SC_ix_iy;
  TH2F *hEB_SC_ieta_iphi;
  int CurrentEvent;
  float EnergyThreshold;
  int theEvent;
  int FirstEvent;
  int LastEvent;

};

#endif
