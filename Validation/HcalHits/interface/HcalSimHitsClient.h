#ifndef ValidationHcalSimHitsClient_H
#define ValidationHcalSimHitsClient_H

// -*- C++ -*-
//
// 
/*
 Description: This is a SImHit CLient code
*/

//
// Originally create by: Bhawna Gomber
//                        
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

class HcalSimHitsClient : public edm::EDAnalyzer {
 
 private:
  DQMStore* dbe_; //dbe seems to be the standard name for this, I dont know why. We of course dont own it
  std::string outputFile_;

  edm::ParameterSet conf_;

  bool verbose_;
  bool debug_;
  static const int nType = 25;
  static const int nTime = 4;
  static const int nType1 = 4;
  
  std::string dirName_;

 public:
  explicit HcalSimHitsClient(const edm::ParameterSet& );
  virtual ~HcalSimHitsClient();
  
  virtual void beginJob(void);
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  virtual void runClient_();   

  int SimHitsEndjob(const std::vector<MonitorElement*> &hcalMEs);

};

#endif
