// -*- C++ -*-
//
// Package:    RoadSearchEventFilter
// Class:      RoadSearchEventFilter
// 
/**\class RoadSearchEventFilter RoadSearchEventFilter.cc TrackingTools/RoadSearchEventFilter/src/RoadSearchEventFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Carsten Noeding
//         Created:  Mon Mar 19 13:51:22 CDT 2007
// $Id$
//
//


// system include files
#include <memory>

#include "TrackingTools/RoadSearchEventFilter/interface/RoadSearchEventFilter.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"


RoadSearchEventFilter::RoadSearchEventFilter(const edm::ParameterSet& iConfig)
{
  numberOfSeeds_       = iConfig.getUntrackedParameter<unsigned int>("NumberOfSeeds");
  seedCollectionLabel_ = iConfig.getUntrackedParameter<std::string>("SeedCollectionLabel");

}


RoadSearchEventFilter::~RoadSearchEventFilter()
{
 
}



// ------------ method called on each new Event  ------------
bool
RoadSearchEventFilter::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //using namespace edm;

   bool result = true; 

   const TrajectorySeedCollection *rsSeedCollection = 0;
   try {
     edm::Handle<TrajectorySeedCollection> rsSeedHandle;
     iEvent.getByLabel(seedCollectionLabel_,rsSeedHandle);
     rsSeedCollection = rsSeedHandle.product();
   }
   catch (edm::Exception const& x) {
     if ( x.categoryCode() == edm::errors::ProductNotFound ) {
       if ( x.history().size() == 1 ) {
	 edm::LogWarning("RoadSearchEventFilter") << "Collection reco::TrajectorySeedCollection with label " << seedCollectionLabel_ << " cannot be found.";
	}
      }
    }

   if (rsSeedCollection->size() > numberOfSeeds_) {
     result=false;
     edm::LogWarning("RoadSearch") << "Found " << rsSeedCollection->size() << " seeds -> skip event.";
   }

   return result;

}


void 
RoadSearchEventFilter::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RoadSearchEventFilter::endJob() {
}

