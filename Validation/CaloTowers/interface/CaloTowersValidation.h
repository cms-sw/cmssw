#ifndef CaloTowersValidation_H
#define CaloTowersValidation_H
 

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

#include <vector>
#include <utility>
#include <ostream>
#include <string>
#include <algorithm>
#include <cmath>

using namespace cms;
using namespace edm;
using namespace std;



class CaloTowersValidation : public edm::EDAnalyzer {
 public:
   CaloTowersValidation(edm::ParameterSet const& conf);
  ~CaloTowersValidation();
  virtual void analyze(edm::Event const& e, edm::EventSetup const& c);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

 private:
  
  DaqMonitorBEInterface* dbe_;
  string outputFile_;


  string theCaloTowerCollectionLabel;


  MonitorElement* meEnergyHcalvsEcal;
  MonitorElement* meEnergyHO; 
  MonitorElement* meEnergyEcal; 
  MonitorElement* meEnergyHcal; 
  MonitorElement* meNumFiredTowers;

  MonitorElement* meEnergyEcalTower;
  MonitorElement* meEnergyHcalTower;
  MonitorElement* meTotEnergy;
};

#endif
