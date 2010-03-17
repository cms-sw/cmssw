// -*- C++ -*-
//
// Package:    TrackTriggerHitProducer
// Class:      TrackTriggerHitProducer
// 
/**\class TrackTriggerHitProducer TrackTriggerHitProducer.cc L1Trigger/TrackTriggerHitProducer/src/TrackTriggerHitProducer.cc

 Description: Creates TrackTriggerHits by copying them from the PixelDigis while applying a threshold on adc.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Steve Stroiney
//         Created:  Wed Oct 15 12:09:57 EDT 2008
// $Id: TrackTriggerHitProducer.cc,v 1.2 2010/02/03 09:46:37 arose Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "SimDataFormats/SLHC/interface/TrackTriggerHit.h"

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometryRecord.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetUnit.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerDetId.h"

#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"

using namespace std;
using namespace edm;

//
// class decleration
//

class TrackTriggerHitProducer : public edm::EDProducer {
   public:
      explicit TrackTriggerHitProducer(const edm::ParameterSet&);
      ~TrackTriggerHitProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      edm::InputTag input_;
      
      unsigned int threshold_; // Only hits with adc >= threshold_ will be kept.
      
      ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
      const cmsUpgrades::StackedTrackerGeometry *theStackedTrackers;
      cmsUpgrades::StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;

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
TrackTriggerHitProducer::TrackTriggerHitProducer(const edm::ParameterSet& iConfig):
  input_( iConfig.getParameter<edm::InputTag>( "input" ) ),
  threshold_( iConfig.getParameter<unsigned int>( "threshold" ) )
{
   //register your products
/* Examples
   produces<ExampleData2>();

   //if do put with a label
   produces<ExampleData2>("label");
*/
	produces< edm::DetSetVector<TrackTriggerHit> >();

   //now do what ever other initialization is needed
  
}


TrackTriggerHitProducer::~TrackTriggerHitProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackTriggerHitProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

	Handle<DetSetVector<PixelDigi> > theDigis;
	iEvent.getByLabel(input_, theDigis);

// container to hold the hits, will be inserted into event
	std::auto_ptr< DetSetVector<TrackTriggerHit> > hitsOutput( new DetSetVector<TrackTriggerHit>() );
	
// create the TrackTriggerHits here
        for (	StackedTrackerIterator = theStackedTrackers->stacks().begin(); 
				StackedTrackerIterator != theStackedTrackers->stacks().end(); 
				++StackedTrackerIterator
	    	){
            cmsUpgrades::StackedTrackerDetId Id = (**StackedTrackerIterator).Id() ;

			cmsUpgrades::StackedTrackerDetUnit::StackContents theMemberIds = (**StackedTrackerIterator).theStackMembers();
			cmsUpgrades::StackedTrackerDetUnit::StackContentsIterator theMemberIdIter;

//			cout << "theId:" << Id.layer() << "." << Id.iPhi() << "." << Id.iZ() << endl;

			for(	theMemberIdIter = theMemberIds.begin();
					theMemberIdIter != theMemberIds.end();
					++theMemberIdIter
				){
//					cout << "\ttheMemberId:" << PXBDetId(theMemberIdIter->second).layer() 
//									  << "." << PXBDetId(theMemberIdIter->second).ladder()
//									  << "." << PXBDetId(theMemberIdIter->second).module()
//									  << endl;

					if( theDigis->find(theMemberIdIter->second) != theDigis->end() ){
						const DetSet<PixelDigi>& thisDetSet = *(theDigis->find(theMemberIdIter->second)); //*theDigis_itr;
//						cout << "\t\tsize:" << thisDetSet.size() << endl;

						DetSet<TrackTriggerHit> hitsForThisDetId( theMemberIdIter->second );
				
						for ( DetSet<PixelDigi>::const_iterator thisDetSet_itr = thisDetSet.begin(); thisDetSet_itr != thisDetSet.end(); thisDetSet_itr++ )
						{
							const PixelDigi& thisPixelDigi = *thisDetSet_itr;
			
							if ( thisPixelDigi.adc() >= threshold_ )
							{
								hitsForThisDetId.push_back( TrackTriggerHit(thisPixelDigi.row(), thisPixelDigi.column()) );
							}
						}
			
						if ( !(hitsForThisDetId.empty()) ) hitsOutput->insert( hitsForThisDetId );

					}
				}

		/*
		// Skip if this DetId does not correspond to a part of the track trigger.
		if ( !(triggerDetectorInfo_.isTrackTrigger( thisDetSet.detId() )) ) continue;
		
		DetSet<TrackTriggerHit> hitsForThisDetId(thisDetSet.detId());
		
		for ( DetSet<PixelDigi>::const_iterator thisDetSet_itr = thisDetSet.begin(); thisDetSet_itr != thisDetSet.end(); thisDetSet_itr++ )
		{
			const PixelDigi& thisPixelDigi = *thisDetSet_itr;
			
			if ( thisPixelDigi.adc() >= threshold_ )
			{
				hitsForThisDetId.push_back( TrackTriggerHit(thisPixelDigi.row(), thisPixelDigi.column()) );
			}
		}
		
		if ( !(hitsForThisDetId.empty()) ) hitsOutput->insert( hitsForThisDetId );
*/
	}
	

// insert the hits
	iEvent.put(hitsOutput);

}

// ------------ method called once each job just before starting event loop  ------------
void 
TrackTriggerHitProducer::beginJob( const edm::EventSetup& iSetup )
{
        iSetup.get<cmsUpgrades::StackedTrackerGeometryRecord>().get(StackedTrackerGeomHandle);
        theStackedTrackers = StackedTrackerGeomHandle.product();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackTriggerHitProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackTriggerHitProducer);

