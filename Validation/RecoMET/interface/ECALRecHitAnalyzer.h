#ifndef ECALRECHITANALYZER_H
#define ECALRECHITANALYZER_H

// author: Bobby Scurlock (The University of Florida)
// date: 11/20/2006

#include <memory>
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DQMServices/Core/interface/DQMStore.h"

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

#include <memory>
#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <TLorentzVector.h>
#include <string>
#include <map>

#include <TH1.h>
#include <TH2.h>
#include <TFile.h>
#include <TMath.h>
#include "DQMServices/Core/interface/MonitorElement.h"

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
  // DAQ Tools
  DQMStore* dbe_;
  std::map<std::string, MonitorElement*> me;
  // Inputs from Configuration
  std::string outputFile_;
  std::string geometryFile_;
  edm::InputTag EBRecHitsLabel_;
  edm::InputTag EERecHitsLabel_;
  bool debug_;
  bool dumpGeometry_;

  int CurrentEvent;
};

#endif
