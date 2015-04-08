#ifndef ValidationHGCalDigiClient_H
#define ValidationHGCalDigiClient_H

// -*- C++ -*-
/*
 Description: This is a SImHit CLient code
*/
//
// Originally create by: Kalyanmoy Chatterjee
//       and Raman Khurana
//                        


#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <unistd.h>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"

class DQMStore;
class MonitorElement;

class HGCalDigiClient : public edm::EDAnalyzer {
 
private:
  DQMStore*   dbe_;
  std::string outputFile_;
  std::string nameDetector_;
  int         verbosity_;
  int         layers_;

public:
  explicit HGCalDigiClient(const edm::ParameterSet& );
  virtual ~HGCalDigiClient();
  
  virtual void beginJob(void);
  virtual void endJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void endRun(const edm::Run& run, const edm::EventSetup& c);
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, const edm::EventSetup& c);
  void         runClient_();   
  int          digisEndjob(const std::vector<MonitorElement*> &hcalMEs);
};

#endif
