#ifndef HCALDIGITESTER_H
#define HCALDIGITESTER_H

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include <map>
#include "Validation/HcalDigis/src/HcalSubdetDigiMonitor.h"

class HcalDigiTester : public edm::EDAnalyzer {
public:
  explicit HcalDigiTester(const edm::ParameterSet&);
  ~HcalDigiTester();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  template<class Digi>  void reco(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;  

 private:

  double dR(double eta1, double phi1, double eta2, double phi2);
  void eval_occupancy();

  // choose the correct subdet
  HcalSubdetDigiMonitor * monitor();

  DQMStore* dbe_;
  
  edm::InputTag inputTag_;
  std::string outputFile_;
  std::string hcalselector_;
  std::string zside_;
  std::string mode_;
  int noise_;             
// flag to distinguish between 
                          // particular subdet only case and "global" noise one

  edm::ESHandle<CaloGeometry> geometry ;
  edm::ESHandle<HcalDbService> conditions;
  float pedvalue;
  int nevent1;
  int nevent2;
  int nevent3;
  int nevent4;
  int nevtot;
  std::map<std::string, HcalSubdetDigiMonitor*> monitors_;

};

#endif

