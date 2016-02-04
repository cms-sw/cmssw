

#ifndef PFTESTER_H
#define PFTESTER_H

// author: Mike Schmitt (The University of Florida)
// date: 11/7/2007

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <map>

class PFTester: public edm::EDAnalyzer {
public:

  explicit PFTester(const edm::ParameterSet&);
  virtual ~PFTester();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob() ;
  virtual void endJob() ;

 private:

  // DAQ Tools
  DQMStore* dbe_;
  std::map<std::string, MonitorElement*> me;

  // Inputs from Configuration File
  std::string outputFile_;
  std::string inputPFlowLabel_;

};

#endif // PFTESTER_H
