#ifndef EcalPreshowerRecHitsValidation_H
#define EcalPreshowerRecHitsValidation_H

/*
 * \file EcalPreshowerRecHitsValidation.h
 *
 * $Date: 2008/02/29 20:48:30 $
 * \author C. Rovelli
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"

class EcalPreshowerRecHitsValidation: public edm::EDAnalyzer{
  
 public:
  
  /// Constructor
  EcalPreshowerRecHitsValidation(const edm::ParameterSet& ps);
  
  /// Destructor
  ~EcalPreshowerRecHitsValidation();
  
 protected:
  
  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  
  // BeginJob
  void beginJob();
  
  // EndJob
  void endJob(void);
  
 private:
  
  bool verbose_;
  
  DQMStore* dbe_;

  edm::InputTag EEuncalibrechitCollection_;
  edm::InputTag EErechitCollection_;
  edm::InputTag ESrechitCollection_;
  
  MonitorElement* meESRecHitsEnergy_;    
  MonitorElement* meESRecHitsEnergy_zp1st_;
  MonitorElement* meESRecHitsEnergy_zp2nd_;
  MonitorElement* meESRecHitsEnergy_zm1st_;
  MonitorElement* meESRecHitsEnergy_zm2nd_;
  MonitorElement* meESRecHitsMultip_;
  MonitorElement* meESRecHitsMultip_zp1st_;
  MonitorElement* meESRecHitsMultip_zp2nd_;
  MonitorElement* meESRecHitsMultip_zm1st_;
  MonitorElement* meESRecHitsMultip_zm2nd_;
  MonitorElement* meESEERecHitsEnergy_zp_;
  MonitorElement* meESEERecHitsEnergy_zm_;
  MonitorElement* meESRecHitsStripOccupancy_zp1st_[36];
  MonitorElement* meESRecHitsStripOccupancy_zm1st_[36];
  MonitorElement* meESRecHitsStripOccupancy_zp2nd_[36];
  MonitorElement* meESRecHitsStripOccupancy_zm2nd_[36];

};

#endif
