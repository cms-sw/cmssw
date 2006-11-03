//#ifndef HCALDIGITESTER_H
//#define HCALDIGITESTER_H

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"
#include <map>
#include "Validation/HcalDigis/src/HcalSubdetDigiMonitor.h"

class HcalDigiTester : public edm::EDAnalyzer {
public:
  explicit HcalDigiTester(const edm::ParameterSet&);
  ~HcalDigiTester();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
 template<class Digi>  void reco(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
 

 private:
  // choose the correct subdet
  HcalSubdetDigiMonitor * monitor();

  DaqMonitorBEInterface* dbe_;
  
  std::string outputFile_;
  std::string hcalselector_;
  bool subpedvalue_;

  edm::ESHandle<CaloGeometry> geometry ;
  edm::ESHandle<HcalDbService> conditions;
  float pedvalue;

  std::map<std::string, HcalSubdetDigiMonitor*> monitors_;
};

