#ifndef HCALVALIDATION_CALOTOWERS_HCALRECHITSCLIENT
#define HCALVALIDATION_CALOTOWERS_HCALRECHITSCLIENT

// -*- C++ -*-
//
// 
/*
 Description: This is a RecHits client meant to plot rechits quantities 
*/

//
// Originally create by: Hongxuan Liu 
//                        May 2010
//

#include <memory>
#include <unistd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include <iostream>
#include <fstream>
#include <vector>

class DQMStore;
class MonitorElement;

class HcalRecHitsClient : public edm::EDAnalyzer {
 
 private:
  DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it
  std::string outputFile_;

  edm::ParameterSet conf_;

  bool verbose_;
  bool debug_;
  bool doSLHC_;

  std::string dirName_;
  std::string dirNameJet_;
  std::string dirNameMET_;

 public:
  explicit HcalRecHitsClient(const edm::ParameterSet& );
  virtual ~HcalRecHitsClient();
  
  virtual void beginJob(void);
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  virtual void runClient_();   

  int HcalRecHitsEndjob(const std::vector<MonitorElement*> &hcalMEs);

};
 
#endif
