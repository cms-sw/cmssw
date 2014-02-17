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
// $Id: TrajectoryFilterAnalyzer.cc,v 1.3 2010/10/03 17:30:57 elmer Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
//#include "RecoTracker/Record/interface/TrajectoryFilterRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class TrajectoryFilterAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TrajectoryFilterAnalyzer(const edm::ParameterSet&);
      ~TrajectoryFilterAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  std::vector<std::string> filterNames;
};

TrajectoryFilterAnalyzer::TrajectoryFilterAnalyzer(const edm::ParameterSet& iConfig)
{  filterNames = iConfig.getParameter<std::vector<std::string> >("filterNames");}


TrajectoryFilterAnalyzer::~TrajectoryFilterAnalyzer(){}

void TrajectoryFilterAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   
   ESHandle<TrajectoryFilter> pTF;
   edm::LogInfo("TrajectoryFilterAnalyzer")<<" I am happy to try and get: "<<filterNames.size()
					   <<" TrajectoryFilter from TrajectoryFilterRecord";
   for (unsigned int i =0; i!= filterNames.size();i++){
     iSetup.get<TrajectoryFilter::Record>().get(filterNames[i], pTF);
     edm::LogInfo("TrajectoryFilterAnalyzer")<<"I was able to create: "<<filterNames[i]
					     <<"\nof type: "<<pTF->name();
   }

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
