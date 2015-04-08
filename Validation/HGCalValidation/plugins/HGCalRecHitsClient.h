#ifndef ValidationHGCalRecHitsClient_H
#define ValidationHGCalRecHitsClient_H

// -*- C++ -*-
/*
 Description: This is a HGCRecHit CLient code
*/
//
// Originally create by: Kalyanmoy Chatterjee
//       and Raman Khurana
//                        

#include <memory>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/LorentzVector.h"

class DQMStore;
class MonitorElement;

class HGCalRecHitsClient : public edm::EDAnalyzer {
 
private:
  DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it
  std::string outputFile_;
  edm::ParameterSet conf_;

  int verbosity_;

  //member data
  std::string dirName_;
  std::string nameDetector_;
  unsigned int layers_;

public:
  explicit HGCalRecHitsClient(const edm::ParameterSet& );
  virtual ~HGCalRecHitsClient();
  
  virtual void beginJob(void);
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  virtual void runClient_();   

  int recHitsEndjob(const std::vector<MonitorElement*> &hcalMEs);
};

#endif
