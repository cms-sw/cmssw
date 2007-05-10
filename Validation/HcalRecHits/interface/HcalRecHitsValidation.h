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
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
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
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

 private:
  
  DaqMonitorBEInterface* dbe_;
  
  std::string outputFile_;
  std::string hcalselector_;
  std::string noiseflag;
  // HE Monitor Elements

  // number of rechits in event 
  MonitorElement* meNumRecHits;

  // number of rechits above threshold 1GEV
  MonitorElement* meNumRecHitsThresh;

  // number of rechits in the cone
  MonitorElement* meNumRecHitsCone;

  // time?
  MonitorElement* meTime;

  // energy of rechits
  MonitorElement* meRecHitsEnergy;


  MonitorElement* me2D;
  
  MonitorElement*  meSumRecHitsEnergy;
  MonitorElement* meSumRecHitsEnergyCone;
  MonitorElement* meEcalHcalEnergy;
 
  
  MonitorElement* meEcalHcalEnergyCone; 
 

  // Histo for eta of each rechit 
  MonitorElement* meEta;

  // Histo for phi of each rechit 
  MonitorElement* mePhi;


  MonitorElement* me2Dprofile;

  // Histo for 2D plot of eta and phi for each depth.
  MonitorElement* meEtaPhiDepth0;
  MonitorElement* meEtaPhiDepth1;
  MonitorElement* meEtaPhiDepth2;
  MonitorElement* meEtaPhiDepth3;
  MonitorElement* meEtaPhiDepth4;

  // Histo (2D plot) for sum of RecHits vs SimHits (hcal only)
  MonitorElement* meRecHitSimHit;
  // profile histo (2D plot) for sum of RecHits vs SimHits (hcal only)
  MonitorElement* meRecHitSimHitProfile;
  // 2D plot of sum of RecHits in HCAL as function of ECAL's one
  MonitorElement* meEnergyHcalVsEcal;
  
  MonitorElement* meRecHitsEnergyNoise;
  // number of ECAL's rechits in cone 0.3 
  MonitorElement* meNumEcalRecHitsCone;


  MonitorElement* e_hb;
  MonitorElement*e_he;
  MonitorElement*e_ho;
  MonitorElement*e_hfl;
  MonitorElement*e_hfs;

  edm::ESHandle<CaloGeometry> geometry ;

};

#endif
