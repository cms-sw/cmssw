
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef GLOBAL_STUB_BUILDER_H
#define GLOBAL_STUB_BUILDER_H

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

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometryRecord.h"

#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

template<	typename T	>
class GlobalStubBuilder : public edm::EDProducer {

	typedef cmsUpgrades::LocalStub< T >							LocalStubType;
	typedef std::vector	< LocalStubType >						LocalStubCollectionType;
	//typedef edm::Ref< LocalStubCollectionType , LocalStubType >	LocalStubRefType;
	typedef edm::Ptr< LocalStubType >	LocalStubPtrType;
	typedef std::vector<  cmsUpgrades::GlobalStub< T > > GlobalStubCollectionType;

	public:
		explicit GlobalStubBuilder(const edm::ParameterSet& iConfig): mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) )
		{
    		produces< GlobalStubCollectionType >();
    		LocalStubsInputTag  = iConfig.getParameter<edm::InputTag>("LocalStubs");
		}

      ~GlobalStubBuilder(){}

	private:
		virtual void beginJob(const edm::EventSetup& iSetup)
		{
			edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
			iSetup.get<cmsUpgrades::StackedTrackerGeometryRecord>().get(StackedTrackerGeomHandle);
			theStackedTracker = StackedTrackerGeomHandle.product();
		}


		virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
		{
			edm::Handle< LocalStubCollectionType > LocalStubHandle;
			iEvent.getByLabel( LocalStubsInputTag , LocalStubHandle);

			std::auto_ptr< GlobalStubCollectionType > GlobalStubsForOutput(new GlobalStubCollectionType );

			for (  unsigned int i = 0; i != LocalStubHandle->size() ; ++i ) {
				GlobalPoint innerHitPosition = LocalStubHandle->at(i).averagePosition( &(*theStackedTracker) , 0);
				GlobalPoint outerHitPosition = LocalStubHandle->at(i).averagePosition( &(*theStackedTracker) , 1);

				GlobalPoint globalPosition( (innerHitPosition.x()+outerHitPosition.x())/2 , (innerHitPosition.y()+outerHitPosition.y())/2 , (innerHitPosition.z()+outerHitPosition.z())/2 );
				GlobalVector directionVector(  outerHitPosition.x()-innerHitPosition.x()    ,  outerHitPosition.y()-innerHitPosition.y()    ,  outerHitPosition.z()-innerHitPosition.z()    );

				//cmsUpgrades::GlobalStub<T> tempGlobalStub( LocalStubRefType(LocalStubHandle,i) , globalPosition , directionVector );
				cmsUpgrades::GlobalStub<T> tempGlobalStub( LocalStubPtrType(LocalStubHandle,i) , globalPosition , directionVector );
				GlobalStubsForOutput->push_back( tempGlobalStub );
			}

			std::cout	<<"Made " << GlobalStubsForOutput->size() << " global stubs of type " << (mClassInfo->TemplateTypes().begin()->second) << "." << std::endl;
			edm::OrphanHandle<GlobalStubCollectionType> GlobalStubHandle = iEvent.put(GlobalStubsForOutput);

		}

      virtual void endJob(){}

      
      // ----------member data ---------------------------
		const cmsUpgrades::StackedTrackerGeometry *theStackedTracker;
		cmsUpgrades::StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;

        edm::InputTag LocalStubsInputTag;

		const cmsUpgrades::classInfo *mClassInfo;

};

#endif

