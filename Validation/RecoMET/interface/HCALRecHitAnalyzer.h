#ifndef HCALRECHITANALYZER_H
#define HCALRECHITANALYZER_H

// author: Bobby Scurlock (The University of Florida)
// date: 8/24/2006
// modification: Mike Schmitt
// date: 02.28.2007
// note: code rewrite

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include <string>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"

class HCALRecHitAnalyzer: public edm::EDAnalyzer {
public:

  explicit HCALRecHitAnalyzer(const edm::ParameterSet&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&);
  virtual void endJob();

private:

  // DAQ Tools
  DQMStore* dbe_;
  std::map<std::string, MonitorElement*> me;

  // Inputs from Configuration
  std::string outputFile_;
  std::string geometryFile_;
  std::string hBHERecHitsLabel_;
  std::string hFRecHitsLabel_;
  std::string hORecHitsLabel_;
  bool debug_;
  bool dumpGeometry_;

  // Helper Functions
  void FillGeometry(const edm::EventSetup&);
  void DumpGeometry();

  int Nevents;
};

#endif
