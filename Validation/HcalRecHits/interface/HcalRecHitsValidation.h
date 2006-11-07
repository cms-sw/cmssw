#ifndef HcalRecHitsValidation_H
#define HcalRecHitsValidation_H
 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"


#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"
#include "FWCore/Framework/interface/Selector.h"
#include <iostream>


#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimG4Core/Watcher/interface/SimWatcherFactory.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalSourcePositionData.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"


#include <DataFormats/EcalDetId/interface/EBDetId.h>
#include <DataFormats/EcalDetId/interface/EEDetId.h>
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

//#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
//#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"


#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"


#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>

using namespace cms;
using namespace edm;
using namespace std;



class HcalRecHitsValidation : public edm::EDAnalyzer {
 public:
   HcalRecHitsValidation(edm::ParameterSet const& conf);
  ~HcalRecHitsValidation();
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

 private:
  
  DaqMonitorBEInterface* dbe_;
  
  string outputFile_;
  string hcalselector_;
  string noiseflag;
  // HE Monitor Elements
  MonitorElement* meNumRecHits;
  MonitorElement* meNumRecHitsThresh;
  MonitorElement* meNumRecHitsCone;
  MonitorElement* meTime;
  MonitorElement* meRecHitsEnergy;
  MonitorElement* me2D;
  MonitorElement*  meSumRecHitsEnergy;
  MonitorElement* meSumRecHitsEnergyCone;
  MonitorElement* meEcalHcalEnergy;
 
  
  MonitorElement* meEcalHcalEnergyCone; 
 
 
  MonitorElement* meEta;
  MonitorElement* mePhi;
  MonitorElement* meDist;
  MonitorElement* me2Dprofile;
  MonitorElement* meEtaPhiDepth0;
  MonitorElement* meEtaPhiDepth1;
  MonitorElement* meEtaPhiDepth2;
  MonitorElement* meEtaPhiDepth3;
  MonitorElement* meEtaPhiDepth4;

  MonitorElement* meRecHitSimHit;
  MonitorElement* meRecHitSimHitProfile;
  MonitorElement* meEnergyHcalVsEcal;
  MonitorElement* meRecHitsEnergyNoise;
  MonitorElement* meNumEcalRecHitsCone;

  ESHandle<CaloGeometry> geometry ;

};

#endif
