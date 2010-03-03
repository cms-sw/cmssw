
/*********************************/
/*********************************/
/**                             **/
/** Stacked Tracker Simulations **/
/**        Andrew W. Rose       **/
/**             2008            **/
/**                             **/
/*********************************/
/*********************************/

#ifndef TRACKLET_BUILDER_H
#define TRACKLET_BUILDER_H

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

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "RecoTracker/TkSeedGenerator/interface/FastHelix.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/MaterialEffects/interface/PropagatorWithMaterial.h"

#include "SLHCUpgradeSimulations/Utilities/interface/classInfo.h"

template< typename T >
class TrackletBuilder : public edm::EDProducer {

	typedef cmsUpgrades::GlobalStub< T > 							GlobalStubType;
	typedef std::vector	< GlobalStubType >							GlobalStubCollectionType;
	//typedef edm::Ref< GlobalStubCollectionType, GlobalStubType >	GlobalStubRefType;
	typedef edm::Ptr< GlobalStubType >	GlobalStubPtrType;
	typedef std::vector<  cmsUpgrades::Tracklet< T > > 				TrackletCollectionType;

	//just for internal use
	//typedef std::map< unsigned int, std::vector< GlobalStubRefType > > 	GlobalStubMapType;
	typedef std::map< unsigned int, std::vector< GlobalStubPtrType > > 	GlobalStubMapType;

	public:
		explicit TrackletBuilder(const edm::ParameterSet& iConfig): mClassInfo( new cmsUpgrades::classInfo(__PRETTY_FUNCTION__) )
		{
			produces<TrackletCollectionType>("ShortTracklets");

			mPtThreshold = iConfig.getParameter<double>("minPtThreshold");
			mIPWidth = iConfig.getParameter<double>("ipWidth");
			mFastPhiCut = iConfig.getParameter<double>("fastPhiCut");
			GlobalStubsInputTag  = iConfig.getParameter<edm::InputTag>("GlobalStubs");
		}

		~TrackletBuilder(){}

	private:

/*		bool CheckTwoStubsForCompatibility( const cmsUpgrades::Tracklet<T> &aTracklet )
		{
			if ( (aTracklet.stub(0).isNull()) || (aTracklet.stub(1).isNull()) ){
				std::cout	<<"Failure!"<<std::endl;
				return false;
			}

			//std::cout<<"Inner Stub" << (aTracklet.stub(0)->print()) <<std::endl;
			//std::cout<<"Outer Stub" << (aTracklet.stub(1)->print()) <<std::endl;

			double outerPointRadius = aTracklet.stub(1)->position().perp();
  			double innerPointRadius = aTracklet.stub(0)->position().perp();
  			double innerPointZ, outerPointZ;

  			// Check for seed compatibility given a pt cut
  			// Threshold computed from radial location of hits
  			double deltaPhiThreshold = (outerPointRadius - innerPointRadius) * mCompatibilityScalingFactor;  

  			// Rebase the angles in terms of 0-2PI, should
  			// really have been written this way in CMSSW...
//  			if ( innerPointPhi < 0.0 ) innerPointPhi += 2.0 * cmsUpgrades::KGMS_PI;
//  			if ( outerPointPhi < 0.0 ) outerPointPhi += 2.0 * cmsUpgrades::KGMS_PI;

			// Delta phi computed from hit phi locations
  			double outerPointPhi = aTracklet.stub(1)->position().phi();
  			double innerPointPhi = aTracklet.stub(0)->position().phi();

  			double deltaPhi = outerPointPhi - innerPointPhi;
			if (deltaPhi<0) deltaPhi = -deltaPhi;
			while( deltaPhi>2.0 * cmsUpgrades::KGMS_PI ) deltaPhi-=(2.0 * cmsUpgrades::KGMS_PI);


			if ( deltaPhi < deltaPhiThreshold ) {
				//edm::LogInfo("StackedTrackerLocalStubSimBuilder")<<"compatible in phi" << flush;
  			  	innerPointZ = aTracklet.stub(0)->position().z();
  			  	outerPointZ = aTracklet.stub(1)->position().z();
				double positiveZBoundary = (mIPWidth - outerPointZ) * (outerPointRadius - innerPointRadius);
				double negativeZBoundary = -(mIPWidth + outerPointZ) * (outerPointRadius - innerPointRadius);
				double multipliedLocation = (innerPointZ - outerPointZ) * outerPointRadius;

				if ( ( multipliedLocation < positiveZBoundary ) && 	( multipliedLocation > negativeZBoundary ) ){
					return true;
				}else{
					return false;
				}
			}else{
				return false;
			}
			return false;
		}*/


		virtual void beginJob(const edm::EventSetup& iSetup)
		{
			iSetup.get<cmsUpgrades::StackedTrackerGeometryRecord>().get(StackedTrackerGeomHandle);
			theStackedTracker = StackedTrackerGeomHandle.product();

			iSetup.get<IdealMagneticFieldRecord>().get(magnet);
			magnet_ = magnet.product();
			mMagneticFieldStrength = magnet_->inTesla(GlobalPoint(0,0,0)).z();
			// Compute the scaling factor (conversion cm->m, Gev-c factor, magnetic field)
			mCompatibilityScalingFactor = (100.0 * 2.0e+9 * mPtThreshold) / (cmsUpgrades::KGMS_C * mMagneticFieldStrength);
			// Invert so we use multiplication instead of division in the comparison
			mCompatibilityScalingFactor = 1.0 / mCompatibilityScalingFactor;
		}



		virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
		{
			unsigned int max_layer=0;

			edm::Handle< GlobalStubCollectionType > GlobalStubHandle;
			iEvent.getByLabel( GlobalStubsInputTag , GlobalStubHandle);

			GlobalStubMapType GlobalStubs;
			//GlobalStubMapType::const_iterator GlobalStubsIter;

			for (  unsigned int i = 0; i != GlobalStubHandle->size() ; ++i ) {
				unsigned int layer = GlobalStubHandle->at(i).Id().layer();
				//GlobalStubs[ layer ].push_back( GlobalStubRefType( GlobalStubHandle , i ) );
				GlobalStubs[ layer ].push_back( GlobalStubPtrType( GlobalStubHandle , i ) );
				if( layer > max_layer ) max_layer = layer;
			}

			////////////////////////////////////////////////////////////////////////////////////////////
			//First get the global stubs and form short tracklets...
			////////////////////////////////////////////////////////////////////////////////////////////

			std::auto_ptr< TrackletCollectionType > ShortTrackletOutput(new TrackletCollectionType );

			for	( unsigned int innerLayerNum = 0 ; innerLayerNum != max_layer ; ++innerLayerNum ){

				//std::vector< GlobalStubRefType > innerHits, outerHits;
				//typedef typename std::vector< GlobalStubRefType >::iterator VRT_IT;
				std::vector< GlobalStubPtrType > innerHits, outerHits;
				typedef typename std::vector< GlobalStubPtrType >::iterator VRT_IT;

				if ( GlobalStubs.find( innerLayerNum )	!=  GlobalStubs.end() )	innerHits = GlobalStubs.find( innerLayerNum )->second;
				if ( GlobalStubs.find( innerLayerNum+1 )!=  GlobalStubs.end() )	outerHits = GlobalStubs.find( innerLayerNum+1 )->second;

				if( innerHits.size() && outerHits.size() ){
					for( VRT_IT innerHitIter = innerHits.begin() ; innerHitIter != innerHits.end() ; ++innerHitIter ){
						for( VRT_IT outerHitIter = outerHits.begin() ; outerHitIter != outerHits.end() ; ++outerHitIter ){
// -----------------------------------------------------------------------------------------------------------------------
							GlobalPoint inner = (**innerHitIter).position();
							GlobalPoint outer = (**outerHitIter).position();

			  				double deltaPhi = inner.phi()-outer.phi();
							if (deltaPhi<0) deltaPhi = -deltaPhi;
							//while( deltaPhi>2.0 * cmsUpgrades::KGMS_PI ) deltaPhi-=(2.0 * cmsUpgrades::KGMS_PI);
							if (deltaPhi > cmsUpgrades::KGMS_PI) deltaPhi = 2 * cmsUpgrades::KGMS_PI - deltaPhi;

							if( deltaPhi<mFastPhiCut ){ // rough search in phi!
								if( (inner.z()>0&&outer.z()>-mIPWidth) || (inner.z()<0&&outer.z()<+mIPWidth)  ){  //rough search by z sector 				

									double outerPointRadius = outer.perp(); 
						  			double innerPointRadius = inner.perp();
						  			double deltaRadius = outerPointRadius - innerPointRadius;

  									double deltaPhiThreshold = deltaRadius * mCompatibilityScalingFactor;  
									if ( deltaPhi < deltaPhiThreshold ) { // detailer search in phi!
										double positiveZBoundary = (mIPWidth - outer.z()) * deltaRadius;
										double negativeZBoundary = -(mIPWidth + outer.z()) * deltaRadius;
										double multipliedLocation = (inner.z() - outer.z()) * outerPointRadius;

										if ( ( multipliedLocation < positiveZBoundary ) && 	( multipliedLocation > negativeZBoundary ) ){ // detailer search in z!
											// all agree so make a tracklet!!!
											cmsUpgrades::Tracklet<T> tempShortTracklet;
											tempShortTracklet.addHit( 0 , *innerHitIter );
											tempShortTracklet.addHit( 1 , *outerHitIter );

											double projected_z = outer.z() - ( outerPointRadius * (outer.z()-inner.z()) / deltaRadius );
											tempShortTracklet.addVertex( GlobalPoint(0.0,0.0,projected_z ));
					
											// add tracket into event
											ShortTrackletOutput->push_back( tempShortTracklet );

										} // end detailed search in z
									} //end if(detailed search in phi)

								} //end if(rough search by z-sector)
							} //end if(rough search in phi)

// -----------------------------------------------------------------------------------------------------------------------

							/*cmsUpgrades::Tracklet<T> tempShortTracklet;
							tempShortTracklet.addHit( 0 , *innerHitIter );
							tempShortTracklet.addHit( 1 , *outerHitIter );

							if( CheckTwoStubsForCompatibility( tempShortTracklet ) ){

								GlobalPoint inner = tempShortTracklet.stub(0)->position();
								GlobalPoint outer = tempShortTracklet.stub(1)->position();
						
								double deltaR_=outer.perp()-inner.perp();
								double deltaZ_=outer.z()-inner.z();

								double projected_z = outer.z() - ( outer.perp() * deltaZ_ / deltaR_ );
								tempShortTracklet.addVertex( GlobalPoint(0.0,0.0,projected_z ));


								ShortTrackletOutput->push_back( tempShortTracklet );
							}*/

						}
					}
				}
			}

			std::cout	<<"Made " << ShortTrackletOutput->size() << " short (two stub) tracklets of type " << (mClassInfo->TemplateTypes().begin()->second) << "." << std::endl;
			iEvent.put(ShortTrackletOutput, "ShortTracklets");
					
			GlobalStubs.clear();
		}


		virtual void endJob(){
		}
      

//
// The class members
//
		edm::ESHandle<cmsUpgrades::StackedTrackerGeometry> StackedTrackerGeomHandle;
		const cmsUpgrades::StackedTrackerGeometry *theStackedTracker;
		//cmsUpgrades::StackedTrackerGeometry::StackContainerIterator StackedTrackerIterator;

        edm::InputTag GlobalStubsInputTag;

		edm::ESHandle<MagneticField> magnet;
		const MagneticField *magnet_;
		double mMagneticFieldStrength;
		double mCompatibilityScalingFactor;

		double mPtThreshold;
		double mIPWidth;

		double mFastPhiCut;

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

