#ifndef HCALDIGITESTER_H
#define HCALDIGITESTER_H

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include <map>
#include "Validation/HcalDigis/src/HcalSubdetDigiMonitor.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalDigiCollections.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

class HcalDigiTester : public DQMEDAnalyzer {
public:

  explicit HcalDigiTester(const edm::ParameterSet&);
  ~HcalDigiTester();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  template<class Digi>  void reco(const edm::Event&, const edm::EventSetup&, const edm::EDGetTokenT<edm::SortedCollection<Digi>  >  &);
  virtual void endRun() ;  

  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &);


 private:

  double dR(double eta1, double phi1, double eta2, double phi2);
  void eval_occupancy();

  // choose the correct subdet
  HcalSubdetDigiMonitor * monitor();

  void constructMonitor(DQMStore::IBooker &);

  edm::InputTag inputTag_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_mc_;
  edm::EDGetTokenT<edm::SortedCollection<HBHEDataFrame> > tok_hbhe_;
  edm::EDGetTokenT<edm::SortedCollection<HODataFrame> > tok_ho_;
  edm::EDGetTokenT<edm::SortedCollection<HFDataFrame> > tok_hf_; 

  std::string outputFile_;
  std::string hcalselector_;
  std::string zside_;
  std::string mode_;
  std::string mc_;
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

