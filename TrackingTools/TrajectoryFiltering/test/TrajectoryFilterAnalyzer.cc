// -*- C++ -*-
//
// Package:    TrajectoryFilterAnalyzer
// Class:      TrajectoryFilterAnalyzer
// 
/**\class TrajectoryFilterAnalyzer TrajectoryFilterAnalyzer.cc TrackingTools/TrajectoryFilterAnalyzer/src/TrajectoryFilterAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Thu Oct  4 21:42:40 CEST 2007
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilterFactory.h"

class TrajectoryFilterAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TrajectoryFilterAnalyzer(const edm::ParameterSet&);
      ~TrajectoryFilterAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

    std::vector<std::unique_ptr<TrajectoryFilter>> filters;
};

namespace {
  // only to convert ConsumesCollector&& to ConsumesCollector&
  TrajectoryFilter *createTrajectoryFilter(const std::string& type, const edm::ParameterSet& pset, edm::ConsumesCollector&& iC) {
    return TrajectoryFilterFactory::get()->create(type, pset, iC);
  }
}

TrajectoryFilterAnalyzer::TrajectoryFilterAnalyzer(const edm::ParameterSet& iConfig)
{
  using VPSet = std::vector<edm::ParameterSet>;
  VPSet filterPSets = iConfig.getParameter<std::vector<edm::ParameterSet> >("filters");
  edm::LogInfo("TrajectoryFilterAnalyzer")<<" I am happy to try and get: "<<filterPSets.size()
                                          <<" TrajectoryFilter from TrajectoryFilterFactory";
  for(const auto& pset: filterPSets) {
    const std::string& type = pset.getParameter<std::string>("ComponentType");
    filters.emplace_back(createTrajectoryFilter(type, pset, consumesCollector()));
    edm::LogInfo("TrajectoryFilterAnalyzer")<<"I was able to create: "<< type
                                            <<"\nof type: "<<filters.back()->name();
  }
}


TrajectoryFilterAnalyzer::~TrajectoryFilterAnalyzer(){}

void TrajectoryFilterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


// ------------ method called once each job just before starting event loop  ------------
void 
TrajectoryFilterAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrajectoryFilterAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrajectoryFilterAnalyzer);
