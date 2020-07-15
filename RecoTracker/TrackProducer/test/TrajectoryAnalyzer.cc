// -*- C++ -*-
//
// Package:    TrajectoryAnalyzer
// Class:      TrajectoryAnalyzer
//
/**\class TrajectoryAnalyzer TrajectoryAnalyzer.cc RecoTracker/TrackProducer/test/TrajectoryAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Boris Mangano
//         Created:  Mon Oct 16 10:38:20 CEST 2006
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"

#include <iostream>
// #define COUT(x) edm::LogVerbatim(x)
#define COUT(x) std::cout << x << ' '

//
// class decleration
//

class TrajectoryAnalyzer : public edm::stream::EDAnalyzer<> {
public:
  explicit TrajectoryAnalyzer(const edm::ParameterSet&);
  ~TrajectoryAnalyzer() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::EDGetTokenT<std::vector<Trajectory>> trajTag;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrajectoryAnalyzer::TrajectoryAnalyzer(const edm::ParameterSet& iConfig)
    : trajTag(consumes<std::vector<Trajectory>>(iConfig.getParameter<edm::InputTag>("trajectoryInput"))) {}

TrajectoryAnalyzer::~TrajectoryAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TrajectoryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  Handle<std::vector<Trajectory>> trajCollectionHandle;
  iEvent.getByToken(trajTag, trajCollectionHandle);

  COUT("TrajectoryAnalyzer") << "trajColl->size(): " << trajCollectionHandle->size() << std::endl;
  for (auto it = trajCollectionHandle->begin(); it != trajCollectionHandle->end(); it++) {
    COUT("TrajectoryAnalyzer") << "this traj has " << it->foundHits() << " valid hits"
                               << " , "
                               << "isValid: " << it->isValid() << std::endl;

    auto const& tmColl = it->measurements();
    for (auto itTraj = tmColl.begin(); itTraj != tmColl.end(); itTraj++) {
      if (!itTraj->updatedState().isValid())
        continue;
      COUT("TrajectoryAnalyzer") << "tm number: " << (itTraj - tmColl.begin()) + 1 << " , "
                                 << "tm.backwardState.pt: " << itTraj->backwardPredictedState().globalMomentum().perp()
                                 << " , "
                                 << "tm.forwardState.pt:  " << itTraj->forwardPredictedState().globalMomentum().perp()
                                 << " , "
                                 << "tm.updatedState.pt:  " << itTraj->updatedState().globalMomentum().perp() << " , "
                                 << "tm.globalPos.perp: " << itTraj->updatedState().globalPosition().perp()
                                 << std::endl;
    }
  }
}

//define this as a plug-in

DEFINE_FWK_MODULE(TrajectoryAnalyzer);
