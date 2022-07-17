// -*- C++ -*-
//
// Package:    TrajectoryCleanerAnalyzer
// Class:      TrajectoryCleanerAnalyzer
//
/**\class TrajectoryCleanerAnalyzer TrajectoryCleanerAnalyzer.cc TrackingTools/TrajectoryCleanerAnalyzer/src/TrajectoryCleanerAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Oct 12 03:46:09 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

class TrajectoryCleanerAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit TrajectoryCleanerAnalyzer(const edm::ParameterSet&);
  ~TrajectoryCleanerAnalyzer();

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  std::vector<edm::ESGetToken<TrajectoryCleaner, TrajectoryCleaner::Record>> cleanerTokens;
  std::vector<std::string> cleanerNames;
};

TrajectoryCleanerAnalyzer::TrajectoryCleanerAnalyzer(const edm::ParameterSet& iConfig) {
  cleanerNames = iConfig.getParameter<std::vector<std::string>>("cleanerNames");
  for (unsigned int i = 0; i != cleanerNames.size(); i++) {
    cleanerTokens[i] = esConsumes(edm::ESInputTag{"", cleanerNames[i]});
  }
}

TrajectoryCleanerAnalyzer::~TrajectoryCleanerAnalyzer() = default;

void TrajectoryCleanerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::LogInfo("TrajectoryCleanerAnalyzer")
      << " I am happy to try and get: " << cleanerNames.size() << " TrajectoryFilter from TrajectoryCleaner::Record";
  for (unsigned int i = 0; i != cleanerNames.size(); i++) {
    const TrajectoryCleaner* pTC = &iSetup.getData(cleanerTokens[i]);
    edm::LogInfo("TrajectoryCleanerAnalyzer") << "I was able to create: " << cleanerNames[i] << " pointer: " << pTC;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrajectoryCleanerAnalyzer);
