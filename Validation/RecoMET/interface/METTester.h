#ifndef METTESTER_H
#define METTESTER_H

// author: Mike Schmitt (The University of Florida)
// date: 8/24/2006
// modification: Bobby Scurlock 
// date: 03.11.2006
// note: added RMS(METx) vs SumET capability 
// modification: Rick Cavanaugh
// date: 05.11.2006 
// note: added configuration parameters 
// modification: Mike Schmitt
// date: 02.28.2007
// note: code rewrite

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <map>

class METTester: public edm::EDAnalyzer {
public:

  explicit METTester(const edm::ParameterSet&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

 private:

  // DAQ Tools
  DaqMonitorBEInterface* dbe_;
  std::map<std::string, MonitorElement*> me;

  // Inputs from Configuration File
  std::string outputFile_;
  std::string inputGenMETLabel_;
  std::string inputCaloMETLabel_;

};

#endif // METTESTER_H
