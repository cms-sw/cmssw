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
// $Id: RoadSearchEventFilter.cc,v 1.4 2009/12/14 22:24:25 wmtan Exp $
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
   bool result = true; 

   const TrajectorySeedCollection *rsSeedCollection = 0;
   edm::Handle<TrajectorySeedCollection> rsSeedHandle;

   iEvent.getByLabel(seedCollectionLabel_,rsSeedHandle);

   if( rsSeedHandle.isValid() ){
     rsSeedCollection = rsSeedHandle.product();
   } else {
     throw cms::Exception("CorruptData")
       << "RoadSearchEventFilter requires collection reco::TrajectorySeedCollection with label " << seedCollectionLabel_ << "\n";
   }

   if (rsSeedCollection->size() > numberOfSeeds_) {
     result=false;
     edm::LogError("TooManySeeds") << "Found " << rsSeedCollection->size() << " seeds -> skip event.";
   }

   return result;

}


void 
RoadSearchEventFilter::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
RoadSearchEventFilter::endJob() {
}

