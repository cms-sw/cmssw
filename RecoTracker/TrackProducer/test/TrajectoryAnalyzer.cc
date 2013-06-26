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
// $Id: TrajectoryAnalyzer.cc,v 1.4 2010/02/25 00:33:36 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h"


using namespace std;

//
// class decleration
//


class TrajectoryAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TrajectoryAnalyzer(const edm::ParameterSet&);
      ~TrajectoryAnalyzer();


   private:
      virtual void beginRun(edm::Run & run, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
  edm::ParameterSet param_;
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
{
  param_ = iConfig;

   //now do what ever initialization is needed

}


TrajectoryAnalyzer::~TrajectoryAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TrajectoryAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  
   Handle<vector<Trajectory> > trajCollectionHandle;
   iEvent.getByLabel(param_.getParameter<string>("trajectoryInput"),trajCollectionHandle);
   //iEvent.getByType(trajCollection);
   
   edm::LogVerbatim("TrajectoryAnalyzer") << "trajColl->size(): " << trajCollectionHandle->size() ;
   for(vector<Trajectory>::const_iterator it = trajCollectionHandle->begin(); it!=trajCollectionHandle->end();it++){
     edm::LogVerbatim("TrajectoryAnalyzer") << "this traj has " << it->foundHits() << " valid hits"  << " , "
					    << "isValid: " << it->isValid() ;

     vector<TrajectoryMeasurement> tmColl = it->measurements();
     for(vector<TrajectoryMeasurement>::const_iterator itTraj = tmColl.begin(); itTraj!=tmColl.end(); itTraj++){
       if(! itTraj->updatedState().isValid()) continue;
       edm::LogVerbatim("TrajectoryAnalyzer") << "tm number: " << (itTraj - tmColl.begin()) + 1<< " , "
					      << "tm.backwardState.pt: " << itTraj->backwardPredictedState().globalMomentum().perp() << " , "
					      << "tm.forwardState.pt:  " << itTraj->forwardPredictedState().globalMomentum().perp() << " , "
					      << "tm.updatedState.pt:  " << itTraj->updatedState().globalMomentum().perp()  << " , "
					      << "tm.globalPos.perp: "   << itTraj->updatedState().globalPosition().perp() ;       
     }
   }

}


// ------------ method called once each job just before starting event loop  ------------
void 
TrajectoryAnalyzer::beginRun(edm::Run & run, const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrajectoryAnalyzer::endJob() {
}

//define this as a plug-in

DEFINE_FWK_MODULE(TrajectoryAnalyzer);
