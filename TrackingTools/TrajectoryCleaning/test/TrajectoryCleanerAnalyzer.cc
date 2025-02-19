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
// $Id: TrajectoryCleanerAnalyzer.cc,v 1.3 2010/10/03 17:28:59 elmer Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class TrajectoryCleanerAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TrajectoryCleanerAnalyzer(const edm::ParameterSet&);
      ~TrajectoryCleanerAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  std::vector<std::string> cleanerNames;
};

TrajectoryCleanerAnalyzer::TrajectoryCleanerAnalyzer(const edm::ParameterSet& iConfig)
{cleanerNames= iConfig.getParameter<std::vector<std::string> >("cleanerNames");}

TrajectoryCleanerAnalyzer::~TrajectoryCleanerAnalyzer(){}

void TrajectoryCleanerAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   edm::ESHandle<TrajectoryCleaner> pTC;
   edm::LogInfo("TrajectoryCleanerAnalyzer")<<" I am happy to try and get: "<<cleanerNames.size()
					   <<" TrajectoryFilter from TrajectoryCleaner::Record";
   for (unsigned int i =0; i!= cleanerNames.size();i++){
     iSetup.get<TrajectoryCleaner::Record>().get(cleanerNames[i], pTC);
     edm::LogInfo("TrajectoryCleanerAnalyzer")<<"I was able to create: "<<cleanerNames[i];
   }
}


// ------------ method called once each job just before starting event loop  ------------
void 
TrajectoryCleanerAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrajectoryCleanerAnalyzer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrajectoryCleanerAnalyzer);
