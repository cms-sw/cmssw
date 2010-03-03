
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef TRACKLET_CHAIN_BUILDER_H
#define TRACKLET_CHAIN_BUILDER_H

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

#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometryRecord.h"
#include "SLHCUpgradeSimulations/Utilities/interface/StackedTrackerGeometry.h"
#include "SimDataFormats/SLHC/interface/StackedTrackerTypes.h"

#include "SLHCUpgradeSimulations/Utilities/interface/constants.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

template< typename T >
class TrackletChainBuilder : public edm::EDProducer {

	typedef cmsUpgrades::Tracklet< T > 					ShortTrackletType;
	typedef std::vector	< ShortTrackletType >				ShortTrackletCollectionType;
	typedef edm::Ptr< ShortTrackletType >					ShortTrackletPtrType;
	typedef std::vector<  cmsUpgrades::Tracklet< T > > 			TrackletCollectionType;

	//just for internal use
	//typedef std::map< unsigned int, std::vector< ShortTrackletRefType > > 	ShortTrackletMapType;
	typedef std::map< unsigned int, std::vector< ShortTrackletPtrType > > 	ShortTrackletMapType;

	public:
		explicit TrackletChainBuilder(const edm::ParameterSet& iConfig): mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) )
		{
			produces<TrackletCollectionType>("TrackletChains");

			mPtThreshold = iConfig.getParameter<double>("minPtThreshold");
			mZmatch = iConfig.getParameter<double>("Zmatch");
			ShortTrackletInputTag  = iConfig.getParameter<edm::InputTag>("ShortTracklets");
		}

		~TrackletChainBuilder(){}

	private:
		virtual void beginJob(const edm::EventSetup& iSetup)
		{
			iSetup.get<cmsUpgrades::StackedTrackerGeometryRecord>().get(StackedTrackerGeomHandle);
			theStackedTracker = StackedTrackerGeomHandle.product();
		}

		virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
		{
			unsigned int max_layer=0;

			edm::Handle< ShortTrackletCollectionType > ShortTrackletHandle;
			iEvent.getByLabel( ShortTrackletInputTag, ShortTrackletHandle);

			ShortTrackletMapType ShortTracklets;

			for (  unsigned int i = 0; i != ShortTrackletHandle->size() ; ++i ) {
				unsigned int layer = (ShortTrackletHandle->at(i)).stub(0)->Id().layer();
				ShortTracklets[ layer ].push_back( ShortTrackletPtrType( ShortTrackletHandle , i ) );
				if( layer > max_layer ) max_layer = layer;
			}

			////////////////////////////////////////////////////////////////////////////////////////////
			//First get the global stubs and form short tracklets...
			////////////////////////////////////////////////////////////////////////////////////////////

			std::auto_ptr< TrackletCollectionType > TrackletChainOutput(new TrackletCollectionType );

			for	( unsigned int innerLayerNum = 0 ; innerLayerNum != max_layer ; ++innerLayerNum ){

				std::vector< ShortTrackletPtrType > innerTracklets, outerTracklets;
				typedef typename std::vector< ShortTrackletPtrType >::iterator VRT_IT;

				if ( ShortTracklets.find( innerLayerNum )  !=  ShortTracklets.end() )	innerTracklets = ShortTracklets.find( innerLayerNum )->second;
				if ( ShortTracklets.find( innerLayerNum+1 )!=  ShortTracklets.end() )	outerTracklets = ShortTracklets.find( innerLayerNum+1 )->second;

				if( innerTracklets.size() && outerTracklets.size() ){
					for( VRT_IT innerTrackletIter = innerTracklets.begin() ; innerTrackletIter != innerTracklets.end() ; ++innerTrackletIter ){
						for( VRT_IT outerTrackletIter = outerTracklets.begin() ; outerTrackletIter != outerTracklets.end() ; ++outerTrackletIter ){
// -----------------------------------------------------------------------------------------------------------------------
							if( (**innerTrackletIter).stub(1)->localStub() == (**outerTrackletIter).stub(0)->localStub() ){
								GlobalPoint P0 = (**innerTrackletIter).stub(0)->position();
								GlobalPoint P1 = (**outerTrackletIter).stub(0)->position(); //which is implicitly = innerTrackletIter->stub(1)->position();
								GlobalPoint P2 = (**outerTrackletIter).stub(1)->position();
		
								double RSF = (P2.perp() - P1.perp()) / (P1.perp() - P0.perp());
								double z2prime = P1.z() + ( (P1.z()-P0.z()) * RSF );
								double z0prime = P1.z() - ( (P2.z()-P1.z()) / RSF );

								if( (fabs(z2prime-P2.z())<mZmatch) || (fabs(z0prime-P0.z())<mZmatch)){
									FastHelix omi(	P2, P1, P0, iSetup);
									double tracklet_pt	=  omi.stateAtVertex().momentum().perp();
									if(tracklet_pt>mPtThreshold){
	
										cmsUpgrades::Tracklet< T > tempTrackletChain;
										tempTrackletChain.addHit( 0 , (**innerTrackletIter).stub(0) );
										tempTrackletChain.addHit( 1 , (**outerTrackletIter).stub(0) );//which is implicitly = innerTrackletIter->stub(1)
										tempTrackletChain.addHit( 2 , (**outerTrackletIter).stub(1) );

										// add tracket into event
										TrackletChainOutput->push_back( tempTrackletChain );
									} 
								} 
							} 

// -----------------------------------------------------------------------------------------------------------------------

						}
					}
				}
			}

			std::cout	<<"Made " << TrackletChainOutput->size() << " tracklet chains of type " << (mClassInfo->TemplateTypes().begin()->second) << "." << std::endl;
			iEvent.put(TrackletChainOutput, "TrackletChains");
					
			ShortTracklets.clear();
		}


		virtual void endJob(){
		}
      

//
// The class members
//
		edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
		const cmsUpgrades::StackedTrackerGeometry *theStackedTracker;
		//cmsUpgrades::StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;

        	edm::InputTag ShortTrackletInputTag;

		double mPtThreshold;
		double mZmatch;

		const cmsUpgrades::classInfo *mClassInfo;

};

//}
//
// constants, enums and typedefs
//


//
// static data member definitions
//

#endif

