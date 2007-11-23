#ifndef PFBENCHMARKANALYZER_H
#define PFBENCHMARKANALYZER_H

// author: Mike Schmitt (The University of Florida)
// date: 11/7/2007

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <map>

class PFBenchmarkAlgo;

class PFBenchmarkAnalyzer: public edm::EDAnalyzer {
public:

  explicit PFBenchmarkAnalyzer(const edm::ParameterSet&);
  virtual ~PFBenchmarkAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

 private:

  // DAQ Tools
  DaqMonitorBEInterface* dbe_;
  std::map<std::string, MonitorElement*> me;

  // PFlow Benchmark Tool
  PFBenchmarkAlgo* algo_;

  // Inputs from Configuration File
  std::string outputFile_;
  std::string inputTruthLabel_;
  std::string inputRecoLabel_;
  std::string benchmarkLabel_;
  bool plotAgainstRecoQuantities_;

};

#endif // PFBENCHMARKANALYZER_H
