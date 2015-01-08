#include <iostream>
#include <fstream>

#include "QuickTrackAssociatorByHitsImpl.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/deltaR.h"

//
// Use the unnamed namespace for utility functions only used in this file
//
namespace
{
	//
	// All of these functions are pretty straightforward but the implementation is type dependent.
	// The templated methods call these for the type specific parts and the compiler will resolve
	// the type and call the correct overload.
	//

	template<class T_element>
	size_t collectionSize( const edm::RefToBaseVector<T_element>& collection )
	{
		return collection.size();
	}

	template<class T_element>
	size_t collectionSize( const edm::Handle<T_element>& hCollection )
	{
		return hCollection->size();
	}

	template<class T_element>
	size_t collectionSize( const edm::RefVector<T_element>& collection )
	{
		return collection.size();
	}

	const reco::Track* getTrackAt( const edm::RefToBaseVector<reco::Track>& trackCollection, size_t index )
	{
		return &*trackCollection[index]; // pretty obscure dereference
	}

	const reco::Track* getTrackAt( const edm::Handle<edm::View<reco::Track> >& pTrackCollection, size_t index )
	{
		return &(*pTrackCollection.product())[index];
	}

	const TrackingParticle* getTrackingParticleAt( const edm::Handle<TrackingParticleCollection>& pCollection, size_t index )
	{
		return &(*pCollection.product())[index];
	}

	const TrackingParticle* getTrackingParticleAt( const edm::RefVector<TrackingParticleCollection>& collection, size_t index )
	{
		return &*collection[index];
	}

	edm::RefToBase<reco::Track> getRefToTrackAt( const edm::RefToBaseVector<reco::Track>& trackCollection, size_t index )
	{
		return trackCollection[index];
	}

	edm::RefToBase<reco::Track> getRefToTrackAt( const edm::Handle<edm::View<reco::Track> >& pTrackCollection, size_t index )
	{
		return edm::RefToBase<reco::Track>( pTrackCollection, index );
	}

	edm::Ref<TrackingParticleCollection> getRefToTrackingParticleAt( const edm::Handle<TrackingParticleCollection>& pCollection, size_t index )
	{
		return edm::Ref<TrackingParticleCollection>( pCollection, index );
	}

	edm::Ref<TrackingParticleCollection> getRefToTrackingParticleAt( const edm::RefVector<TrackingParticleCollection>& collection, size_t index )
	{
		return collection[index];
	}

} // end of the unnamed namespace

QuickTrackAssociatorByHitsImpl::QuickTrackAssociatorByHitsImpl(std::shared_ptr<const TrackerHitAssociator> hitAssoc,
                                                               std::shared_ptr<const ClusterTPAssociationList> clusterToTPMap,

                                                               bool absoluteNumberOfHits,
                                                               double qualitySimToReco,
                                                               double puritySimToReco,
                                                               double cutRecoToSim,
                                                               bool threeHitTracksAreSpecial,
                                                               SimToRecoDenomType simToRecoDenominator):
  hitAssociator_(std::move(hitAssoc)),
  clusterToTPMap_(std::move(clusterToTPMap)),
  qualitySimToReco_(qualitySimToReco),
  puritySimToReco_(puritySimToReco),
  cutRecoToSim_(cutRecoToSim),
  simToRecoDenominator_(simToRecoDenominator) ,
  threeHitTracksAreSpecial_(threeHitTracksAreSpecial),
  absoluteNumberOfHits_(absoluteNumberOfHits)
  {}


reco::RecoToSimCollection QuickTrackAssociatorByHitsImpl::associateRecoToSim( const edm::Handle<edm::View<reco::Track> >& trackCollectionHandle, 
                                                                              const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const
{
	// Only pass the one that was successfully created to the templated method.
	if( not clusterToTPMap_ ) return associateRecoToSimImplementation( trackCollectionHandle, trackingParticleCollectionHandle, *hitAssociator_ );
	else return associateRecoToSimImplementation( trackCollectionHandle, trackingParticleCollectionHandle, *clusterToTPMap_ );
}

reco::SimToRecoCollection QuickTrackAssociatorByHitsImpl::associateSimToReco( const edm::Handle<edm::View<reco::Track> >& trackCollectionHandle, 
                                                                              const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const
{
	// Only pass the one that was successfully created to the templated method.
	if( not clusterToTPMap_ ) return associateSimToRecoImplementation( trackCollectionHandle, trackingParticleCollectionHandle, *hitAssociator_ );
	else return associateSimToRecoImplementation( trackCollectionHandle, trackingParticleCollectionHandle, *clusterToTPMap_ );
}

reco::RecoToSimCollection QuickTrackAssociatorByHitsImpl::associateRecoToSim( const edm::RefToBaseVector<reco::Track>& trackCollection,
                                                                              const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection ) const 
{
	// Only pass the one that was successfully created to the templated method.
	if( not clusterToTPMap_ ) return associateRecoToSimImplementation( trackCollection, trackingParticleCollection, *hitAssociator_ );
	else return associateRecoToSimImplementation( trackCollection, trackingParticleCollection, *clusterToTPMap_ );
}

reco::SimToRecoCollection QuickTrackAssociatorByHitsImpl::associateSimToReco( const edm::RefToBaseVector<reco::Track>& trackCollection, 
                                                                              const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection) const
{
	// Only pass the one that was successfully created to the templated method.
	if( not clusterToTPMap_ ) return associateSimToRecoImplementation( trackCollection, trackingParticleCollection, *hitAssociator_ );
	else return associateSimToRecoImplementation( trackCollection, trackingParticleCollection, *clusterToTPMap_ );
}


template<class T_TrackCollection, class T_TrackingParticleCollection, class T_hitOrClusterAssociator>
reco::RecoToSimCollection QuickTrackAssociatorByHitsImpl::associateRecoToSimImplementation( T_TrackCollection trackCollection, T_TrackingParticleCollection trackingParticleCollection, T_hitOrClusterAssociator hitOrClusterAssociator ) const
{
	reco::RecoToSimCollection returnValue;

	size_t collectionSize=::collectionSize(trackCollection); // Delegate away type specific part

	for( size_t i=0; i < collectionSize; ++i )
	{
		const reco::Track* pTrack=::getTrackAt(trackCollection,i); // Get a normal pointer for ease of use. This part is type specific so delegate.

		// The return of this function has first as the index and second as the number of associated hits
		std::vector < std::pair<edm::Ref<TrackingParticleCollection>,size_t> > trackingParticleQualityPairs=associateTrack( hitOrClusterAssociator, trackingParticleCollection, pTrack->recHitsBegin(), pTrack->recHitsEnd() );

		// int nt = 0;
		for( std::vector<std::pair<edm::Ref<TrackingParticleCollection>,size_t> >::const_iterator iTrackingParticleQualityPair=
				trackingParticleQualityPairs.begin(); iTrackingParticleQualityPair != trackingParticleQualityPairs.end();
				++iTrackingParticleQualityPair )
		{
			const edm::Ref<TrackingParticleCollection>& trackingParticleRef=iTrackingParticleQualityPair->first;
			size_t numberOfSharedHits=iTrackingParticleQualityPair->second;
			size_t numberOfValidTrackHits=pTrack->found();

			if( numberOfSharedHits == 0 ) continue; // No point in continuing if there was no association

			//if electron subtract double counting
			if( abs( trackingParticleRef->pdgId() ) == 11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
			{
				numberOfSharedHits-=getDoubleCount( hitOrClusterAssociator, pTrack->recHitsBegin(), pTrack->recHitsEnd(), trackingParticleRef );
			}

			double quality;
			if( absoluteNumberOfHits_ ) quality=static_cast<double>( numberOfSharedHits );
			else if( numberOfValidTrackHits != 0 ) quality=
					(static_cast<double>( numberOfSharedHits ) / static_cast<double>( numberOfValidTrackHits ));
			else quality=0;
			if( quality > cutRecoToSim_ && !(threeHitTracksAreSpecial_ && numberOfValidTrackHits == 3 && numberOfSharedHits < 3) )
			{
				// Getting the RefToBase is dependent on the type of trackCollection, so delegate that to an overload.
				returnValue.insert( ::getRefToTrackAt(trackCollection,i), std::make_pair( trackingParticleRef, quality ) );
			}
		}
	}
	return returnValue;
}

template<class T_TrackCollection, class T_TrackingParticleCollection, class T_hitOrClusterAssociator>
reco::SimToRecoCollection QuickTrackAssociatorByHitsImpl::associateSimToRecoImplementation( T_TrackCollection trackCollection, T_TrackingParticleCollection trackingParticleCollection, T_hitOrClusterAssociator hitOrClusterAssociator ) const
{
	reco::SimToRecoCollection returnValue;

	size_t collectionSize=::collectionSize(trackCollection); // Delegate away type specific part

	for( size_t i=0; i<collectionSize; ++i )
	{
		const reco::Track* pTrack=::getTrackAt(trackCollection,i); // Get a normal pointer for ease of use. This part is type specific so delegate.

		// The return of this function has first as an edm:Ref to the associated TrackingParticle, and second as the number of associated hits
		std::vector < std::pair<edm::Ref<TrackingParticleCollection>,size_t> > trackingParticleQualityPairs=associateTrack( hitOrClusterAssociator, trackingParticleCollection, pTrack->recHitsBegin(), pTrack->recHitsEnd() );

		// int nt = 0;
		for( std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> >::const_iterator iTrackingParticleQualityPair=trackingParticleQualityPairs.begin();
				iTrackingParticleQualityPair!=trackingParticleQualityPairs.end(); ++iTrackingParticleQualityPair )
		{
			const edm::Ref<TrackingParticleCollection>& trackingParticleRef=iTrackingParticleQualityPair->first;
			size_t numberOfSharedHits=iTrackingParticleQualityPair->second;
			size_t numberOfValidTrackHits=pTrack->found();
			size_t numberOfSimulatedHits=0; // Set a few lines below, but only if required.

			if( numberOfSharedHits==0 ) continue; // No point in continuing if there was no association

			if( simToRecoDenominator_==denomsim || (numberOfSharedHits<3 && threeHitTracksAreSpecial_) ) // the numberOfSimulatedHits is not always required, so can skip counting in some circumstances
			{
				// Note that in the standard TrackAssociatorByHits, all of the hits in associatedTrackingParticleHits are checked for
				// various things.  I'm not sure what these checks are for but they depend on the UseGrouping and UseSplitting settings.
				// This associator works as though both UseGrouping and UseSplitting were set to true, i.e. just counts the number of
				// hits in the tracker.
				numberOfSimulatedHits=trackingParticleRef->numberOfTrackerHits();
			}

			//if electron subtract double counting
			if (abs(trackingParticleRef->pdgId())==11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
			{
				numberOfSharedHits -= getDoubleCount( hitOrClusterAssociator, pTrack->recHitsBegin(), pTrack->recHitsEnd(), trackingParticleRef );
			}

			double purity=static_cast<double>(numberOfSharedHits)/static_cast<double>(numberOfValidTrackHits);
			double quality;
			if( absoluteNumberOfHits_ ) quality=static_cast<double>(numberOfSharedHits);
			else if( simToRecoDenominator_==denomsim && numberOfSimulatedHits != 0 ) quality=static_cast<double>(numberOfSharedHits)/static_cast<double>(numberOfSimulatedHits);
			else if( simToRecoDenominator_==denomreco && numberOfValidTrackHits != 0 ) quality=purity;
			else quality=0;

			if( quality>qualitySimToReco_ && !( threeHitTracksAreSpecial_ && numberOfSimulatedHits==3 && numberOfSharedHits<3 ) && ( absoluteNumberOfHits_ || (purity>puritySimToReco_) ) )
			{
				// Getting the RefToBase is dependent on the type of trackCollection, so delegate that to an overload.
				returnValue.insert( trackingParticleRef, std::make_pair( ::getRefToTrackAt(trackCollection,i), quality ) );
			}
		}
	}
	return returnValue;

}

template<typename T_TPCollection,typename iter> std::vector<std::pair<edm::Ref<TrackingParticleCollection>,size_t> > QuickTrackAssociatorByHitsImpl::associateTrack( const TrackerHitAssociator& hitAssociator, T_TPCollection trackingParticles, iter begin, iter end ) const
{
	// The pairs in this vector have a Ref to the associated TrackingParticle as "first" and the number of associated hits as "second"
	std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > returnValue;

	// The pairs in this vector have first as the sim track identifiers, and second the number of reco hits associated to that sim track.
	// Most reco hits will probably have come from the same sim track, so the number of entries in this vector should be fewer than the
	// number of reco hits.  The pair::second entries should add up to the total number of reco hits though.
	std::vector< std::pair<SimTrackIdentifiers,size_t> > hitIdentifiers=getAllSimTrackIdentifiers( hitAssociator, begin, end );

	// Loop over the TrackingParticles
	size_t collectionSize=::collectionSize(trackingParticles);

	for( size_t i=0; i<collectionSize; ++i )
	{
		const TrackingParticle* pTrackingParticle=getTrackingParticleAt( trackingParticles, i );

		// Ignore TrackingParticles with no hits
		if( pTrackingParticle->numberOfHits()==0 ) continue;

		size_t numberOfAssociatedHits=0;
		// Loop over all of the sim track identifiers and see if any of them are part of this TrackingParticle. If they are, add
		// the number of reco hits associated to that sim track to the total number of associated hits.
		for( std::vector< std::pair<SimTrackIdentifiers,size_t> >::const_iterator iIdentifierCountPair=hitIdentifiers.begin(); iIdentifierCountPair!=hitIdentifiers.end(); ++iIdentifierCountPair )
		{
			if( trackingParticleContainsIdentifier( pTrackingParticle, iIdentifierCountPair->first ) ) numberOfAssociatedHits+=iIdentifierCountPair->second;
		}

		if( numberOfAssociatedHits>0 )
		{
			returnValue.push_back( std::make_pair( getRefToTrackingParticleAt(trackingParticles,i), numberOfAssociatedHits ) );
		}
	}

	return returnValue;
}

template<typename T_TPCollection,typename iter> std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > QuickTrackAssociatorByHitsImpl::associateTrack( const ClusterTPAssociationList& clusterToTPMap, T_TPCollection trackingParticles, iter begin, iter end ) const
{
	// Note that the trackingParticles parameter is not actually required since all the information is in clusterToTPMap,
	// but the method signature has to match the other overload because it is called from a templated method.

	// The pairs in this vector have a Ref to the associated TrackingParticle as "first" and the number of associated clusters as "second"
	// Note: typedef edm::Ref<TrackingParticleCollection> TrackingParticleRef;
	std::vector < std::pair<edm::Ref<TrackingParticleCollection>,size_t> > returnValue;
	if( clusterToTPMap.empty() ) return returnValue;

	// The pairs in this vector have first as the TP, and second the number of reco clusters associated to that TP.
	// Most reco clusters will probably have come from the same sim track (i.e TP), so the number of entries in this
	// vector should be fewer than the number of clusters. The pair::second entries should add up to the total
	// number of reco clusters though.
	std::vector<OmniClusterRef> oClusters=getMatchedClusters( begin, end );

	std::map < TrackingParticleRef, size_t > lmap;
	for( std::vector<OmniClusterRef>::const_iterator it=oClusters.begin(); it != oClusters.end(); ++it )
	{

		std::pair < OmniClusterRef, TrackingParticleRef > clusterTPpairWithDummyTP( *it, TrackingParticleRef() ); //TP is dummy: for clusterTPAssociationListGreater sorting only the cluster is needed
		auto range=std::equal_range( clusterToTPMap.begin(), clusterToTPMap.end(), clusterTPpairWithDummyTP, clusterTPAssociationListGreater );
		if( range.first != range.second )
		{
			for( auto ip=range.first; ip != range.second; ++ip )
			{

				const TrackingParticleRef trackingParticle=(ip->second);

				// Ignore TrackingParticles with no hits
				if( trackingParticle->numberOfHits() == 0 ) continue;

				/* Alternative implementation to avoid the use of lmap... memory slightly improved but slightly slower...
				 std::pair<edm::Ref<TrackingParticleCollection>,size_t> tpIntPair(trackingParticle, 1);
				 auto tp_range = std::equal_range(returnValue.begin(), returnValue.end(), tpIntPair, tpIntPairGreater);
				 if ((tp_range.second-tp_range.first)>1) {
				 edm::LogError("TrackAssociator") << ">>> Error in counting TPs!" << " file: " << __FILE__ << " line: " << __LINE__;
				 }
				 if(tp_range.first != tp_range.second) {
				 tp_range.first->second++;
				 } else {
				 returnValue.push_back(tpIntPair);
				 std::sort(returnValue.begin(), returnValue.end(), tpIntPairGreater);
				 }
				 */
				auto jpos=lmap.find( trackingParticle );
				if( jpos != lmap.end() ) ++jpos->second;
				else lmap.insert( std::make_pair( trackingParticle, 1 ) );
			}
		}
	}
	// now copy the map to returnValue
	for( auto ip=lmap.begin(); ip != lmap.end(); ++ip )
	{
		returnValue.push_back( std::make_pair( ip->first, ip->second ) );
	}
	return returnValue;
}

template<typename iter> std::vector<OmniClusterRef> QuickTrackAssociatorByHitsImpl::getMatchedClusters(iter begin, iter end) const
{
  std::vector<OmniClusterRef> returnValue;
  for (iter iRecHit = begin; iRecHit != end; ++iRecHit) {
    const TrackingRecHit* rhit = getHitFromIter(iRecHit);
    if (rhit->isValid()) {
      int subdetid = rhit->geographicalId().subdetId();
      if (subdetid==PixelSubdetector::PixelBarrel||subdetid==PixelSubdetector::PixelEndcap) {
	const SiPixelRecHit* pRHit = dynamic_cast<const SiPixelRecHit*>(rhit);
	if (!pRHit->cluster().isNonnull())
	  edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!" << " file: " << __FILE__ << " line: " << __LINE__;
	returnValue.push_back(pRHit->omniClusterRef());
      }
      else if (subdetid==SiStripDetId::TIB||subdetid==SiStripDetId::TOB||subdetid==SiStripDetId::TID||subdetid==SiStripDetId::TEC) {
	const std::type_info &tid = typeid(*rhit);
	if (tid == typeid(SiStripMatchedRecHit2D)) {
	  const SiStripMatchedRecHit2D* sMatchedRHit = dynamic_cast<const SiStripMatchedRecHit2D*>(rhit);
	  if (!sMatchedRHit->monoHit().cluster().isNonnull() || !sMatchedRHit->stereoHit().cluster().isNonnull())
	    edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!" << " file: " << __FILE__ << " line: " << __LINE__;
	  returnValue.push_back(sMatchedRHit->monoClusterRef());
	  returnValue.push_back(sMatchedRHit->stereoClusterRef());
	}
	else if (tid == typeid(SiStripRecHit2D)) {
	  const SiStripRecHit2D* sRHit = dynamic_cast<const SiStripRecHit2D*>(rhit);
	  if (!sRHit->cluster().isNonnull())
	    edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!" << " file: " << __FILE__ << " line: " << __LINE__;
	  returnValue.push_back(sRHit->omniClusterRef());
	}
	else if (tid == typeid(SiStripRecHit1D)) {
	  const SiStripRecHit1D* sRHit = dynamic_cast<const SiStripRecHit1D*>(rhit);
	  if (!sRHit->cluster().isNonnull())
	    edm::LogError("TrackAssociator") << ">>> RecHit does not have an associated cluster!" << " file: " << __FILE__ << " line: " << __LINE__;
	  returnValue.push_back(sRHit->omniClusterRef());
	}
	else {
	  auto const & thit = static_cast<BaseTrackerRecHit const&>(*rhit);
	  if ( thit.isProjected() ) {
	  } else {
	    edm::LogError("TrackAssociator") << ">>> getMatchedClusters: TrackingRecHit not associated to any SiStripCluster! subdetid = " << subdetid;
	  }
	}
      }
      else {
	edm::LogError("TrackAssociator") << ">>> getMatchedClusters: TrackingRecHit not associated to any cluster! subdetid = " << subdetid;
      }
    }
  }
  return returnValue;
}

template<typename iter> std::vector< std::pair<QuickTrackAssociatorByHitsImpl::SimTrackIdentifiers,size_t> > QuickTrackAssociatorByHitsImpl::getAllSimTrackIdentifiers( const TrackerHitAssociator& hitAssociator, iter begin, iter end ) const
{
	// The pairs in this vector have first as the sim track identifiers, and second the number of reco hits associated to that sim track.
	std::vector < std::pair<SimTrackIdentifiers,size_t> > returnValue;

	std::vector<SimTrackIdentifiers> simTrackIdentifiers;
	// Loop over all of the rec hits in the track
	//iter tRHIterBeginEnd = getTRHIterBeginEnd( pTrack );
	for( iter iRecHit=begin; iRecHit != end; ++iRecHit )
	{
		if( getHitFromIter( iRecHit )->isValid() )
		{
			simTrackIdentifiers.clear();

			// Get the identifiers for the sim track that this hit came from. There should only be one entry unless clusters
			// have merged (as far as I know).
			hitAssociator.associateHitId( *(getHitFromIter( iRecHit )), simTrackIdentifiers ); // This call fills simTrackIdentifiers
			// Loop over each identifier, and add it to the return value only if it's not already in there
			for( std::vector<SimTrackIdentifiers>::const_iterator iIdentifier=simTrackIdentifiers.begin(); iIdentifier != simTrackIdentifiers.end();
					++iIdentifier )
			{
				std::vector<std::pair<SimTrackIdentifiers,size_t> >::iterator iIdentifierCountPair;
				for( iIdentifierCountPair=returnValue.begin(); iIdentifierCountPair != returnValue.end(); ++iIdentifierCountPair )
				{
					if( iIdentifierCountPair->first.first == iIdentifier->first && iIdentifierCountPair->first.second == iIdentifier->second )
					{
						// This sim track identifier is already in the list, so increment the count of how many hits it relates to.
						++iIdentifierCountPair->second;
						break;
					}
				}
				if( iIdentifierCountPair == returnValue.end() ) returnValue.push_back( std::make_pair( *iIdentifier, 1 ) );
				// This identifier wasn't found, so add it
			}
		}
	}
	return returnValue;
}

bool QuickTrackAssociatorByHitsImpl::trackingParticleContainsIdentifier( const TrackingParticle* pTrackingParticle, const SimTrackIdentifiers& identifier ) const
{
	// Loop over all of the g4 tracks in the tracking particle
	for( std::vector<SimTrack>::const_iterator iSimTrack=pTrackingParticle->g4Track_begin(); iSimTrack != pTrackingParticle->g4Track_end();
			++iSimTrack )
	{
		// And see if the sim track identifiers match
		if( iSimTrack->eventId() == identifier.second && iSimTrack->trackId() == identifier.first )
		{
			return true;
		}
	}

	// If control has made it this far then none of the identifiers were found in
	// any of the g4 tracks, so return false.
	return false;
}

template<typename iter> int QuickTrackAssociatorByHitsImpl::getDoubleCount( const TrackerHitAssociator& hitAssociator, iter startIterator, iter endIterator, TrackingParticleRef associatedTrackingParticle ) const
{
	// This method is largely copied from the standard TrackAssociatorByHits. Once I've tested how much difference
	// it makes I'll go through and comment it properly.

	int doubleCount=0;
	std::vector < SimHitIdpr > SimTrackIdsDC;

	for( iter iHit=startIterator; iHit != endIterator; iHit++ )
	{
		int idcount=0;

		SimTrackIdsDC.clear();
		hitAssociator.associateHitId( *(getHitFromIter( iHit )), SimTrackIdsDC );
		if( SimTrackIdsDC.size() > 1 )
		{
			for( TrackingParticle::g4t_iterator g4T=associatedTrackingParticle->g4Track_begin(); g4T != associatedTrackingParticle->g4Track_end();
					++g4T )
			{
				if( find( SimTrackIdsDC.begin(), SimTrackIdsDC.end(), SimHitIdpr( ( *g4T).trackId(), SimTrackIdsDC.begin()->second ) )
						!= SimTrackIdsDC.end() )
				{
					idcount++;
				}
			}
		}
		if( idcount > 1 ) doubleCount+=(idcount - 1);
	}

	return doubleCount;
}

template<typename iter> int QuickTrackAssociatorByHitsImpl::getDoubleCount( const ClusterTPAssociationList& clusterToTPList, iter startIterator, iter endIterator, TrackingParticleRef associatedTrackingParticle ) const
{
	// This code here was written by Subir Sarkar. I'm just splitting it off into a
	// separate method. - Grimes 01/May/2014

	int doubleCount=0;
	std::vector < SimHitIdpr > SimTrackIdsDC;

	for( iter iHit=startIterator; iHit != endIterator; iHit++ )
	{
		int idcount=0;

		std::vector < OmniClusterRef > oClusters=getMatchedClusters( iHit, iHit + 1 );  //only for the cluster being checked
		for( std::vector<OmniClusterRef>::const_iterator it=oClusters.begin(); it != oClusters.end(); ++it )
		{
			std::pair<OmniClusterRef,TrackingParticleRef> clusterTPpairWithDummyTP( *it, TrackingParticleRef() ); //TP is dummy: for clusterTPAssociationListGreater sorting only the cluster is needed
			auto range=std::equal_range( clusterToTPList.begin(), clusterToTPList.end(), clusterTPpairWithDummyTP, clusterTPAssociationListGreater );
			if( range.first != range.second )
			{
				for( auto ip=range.first; ip != range.second; ++ip )
				{
					const TrackingParticleRef trackingParticle=(ip->second);
					if( associatedTrackingParticle == trackingParticle )
					{
						idcount++;
					}
				}
			}
		}

		if( idcount > 1 ) doubleCount+=(idcount - 1);
	}

	return doubleCount;
}

reco::RecoToSimCollectionSeed QuickTrackAssociatorByHitsImpl::associateRecoToSim( const edm::Handle<edm::View<TrajectorySeed> >& pSeedCollectionHandle_, 
                                                                                  const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const
{

	edm::LogVerbatim( "TrackAssociator" ) << "Starting TrackAssociatorByHitsImpl::associateRecoToSim - #seeds=" << pSeedCollectionHandle_->size()
			<< " #TPs=" << trackingParticleCollectionHandle->size();

	reco::RecoToSimCollectionSeed returnValue;

	size_t collectionSize=pSeedCollectionHandle_->size();

	for( size_t i=0; i < collectionSize; ++i )
	{
		const TrajectorySeed* pSeed= &( *pSeedCollectionHandle_)[i];

		// The return of this function has first as the index and second as the number of associated hits
		std::vector < std::pair<edm::Ref<TrackingParticleCollection>,size_t> > trackingParticleQualityPairs=
				(clusterToTPMap_) ? associateTrack( *clusterToTPMap_, trackingParticleCollectionHandle, pSeed->recHits().first, pSeed->recHits().second ) : associateTrack( *hitAssociator_, trackingParticleCollectionHandle, pSeed->recHits().first, pSeed->recHits().second );
		for( std::vector<std::pair<edm::Ref<TrackingParticleCollection>,size_t> >::const_iterator iTrackingParticleQualityPair=
				trackingParticleQualityPairs.begin(); iTrackingParticleQualityPair != trackingParticleQualityPairs.end();
				++iTrackingParticleQualityPair )
		{
			const edm::Ref<TrackingParticleCollection>& trackingParticleRef=iTrackingParticleQualityPair->first;
			size_t numberOfSharedHits=iTrackingParticleQualityPair->second;
			size_t numberOfValidTrackHits=pSeed->recHits().second - pSeed->recHits().first;

			if( numberOfSharedHits == 0 ) continue; // No point in continuing if there was no association

			//if electron subtract double counting
			if( abs( trackingParticleRef->pdgId() ) == 11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
			{
				if( clusterToTPMap_ ) numberOfSharedHits-=getDoubleCount( *clusterToTPMap_, pSeed->recHits().first, pSeed->recHits().second, trackingParticleRef );
				else numberOfSharedHits-=getDoubleCount( *hitAssociator_, pSeed->recHits().first, pSeed->recHits().second, trackingParticleRef );
			}

			double quality;
			if( absoluteNumberOfHits_ ) quality=static_cast<double>( numberOfSharedHits );
			else if( numberOfValidTrackHits != 0 ) quality=
					(static_cast<double>( numberOfSharedHits ) / static_cast<double>( numberOfValidTrackHits ));
			else quality=0;

			if( quality > cutRecoToSim_ && !(threeHitTracksAreSpecial_ && numberOfValidTrackHits == 3 && numberOfSharedHits < 3) )
			{
				returnValue.insert( edm::RefToBase < TrajectorySeed > (pSeedCollectionHandle_, i), std::make_pair( trackingParticleRef, quality ) );
			}
		}
	}

	LogTrace( "TrackAssociator" ) << "% of Assoc Seeds=" << ((double)returnValue.size()) / ((double)pSeedCollectionHandle_->size());
	returnValue.post_insert();
	return returnValue;

}

reco::SimToRecoCollectionSeed QuickTrackAssociatorByHitsImpl::associateSimToReco( const edm::Handle<edm::View<TrajectorySeed> >& pSeedCollectionHandle_, 
                                                                                  const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const 
{

	edm::LogVerbatim( "TrackAssociator" ) << "Starting TrackAssociatorByHitsImpl::associateSimToReco - #seeds=" << pSeedCollectionHandle_->size()
			<< " #TPs=" << trackingParticleCollectionHandle->size();

	reco::SimToRecoCollectionSeed returnValue;

	size_t collectionSize=pSeedCollectionHandle_->size();

	for( size_t i=0; i < collectionSize; ++i )
	{
		const TrajectorySeed* pSeed= &( *pSeedCollectionHandle_)[i];

		// The return of this function has first as an edm:Ref to the associated TrackingParticle, and second as the number of associated hits
		std::vector < std::pair<edm::Ref<TrackingParticleCollection>,size_t> > trackingParticleQualityPairs=
				(clusterToTPMap_) ? associateTrack( *clusterToTPMap_, trackingParticleCollectionHandle, pSeed->recHits().first, pSeed->recHits().second ) : associateTrack( *hitAssociator_, trackingParticleCollectionHandle, pSeed->recHits().first, pSeed->recHits().second );
		for( std::vector<std::pair<edm::Ref<TrackingParticleCollection>,size_t> >::const_iterator iTrackingParticleQualityPair=
				trackingParticleQualityPairs.begin(); iTrackingParticleQualityPair != trackingParticleQualityPairs.end();
				++iTrackingParticleQualityPair )
		{
			const edm::Ref<TrackingParticleCollection>& trackingParticleRef=iTrackingParticleQualityPair->first;
			size_t numberOfSharedHits=iTrackingParticleQualityPair->second;
			size_t numberOfValidTrackHits=pSeed->recHits().second - pSeed->recHits().first;
			size_t numberOfSimulatedHits=0; // Set a few lines below, but only if required.

			if( numberOfSharedHits == 0 ) continue; // No point in continuing if there was no association

			//if electron subtract double counting
			if( abs( trackingParticleRef->pdgId() ) == 11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
			{
				if( clusterToTPMap_ ) numberOfSharedHits-=getDoubleCount( *clusterToTPMap_, pSeed->recHits().first, pSeed->recHits().second, trackingParticleRef );
				else numberOfSharedHits-=getDoubleCount( *hitAssociator_, pSeed->recHits().first, pSeed->recHits().second, trackingParticleRef );
			}

			if( simToRecoDenominator_ == denomsim || (numberOfSharedHits < 3 && threeHitTracksAreSpecial_) ) // the numberOfSimulatedHits is not always required, so can skip counting in some circumstances
			{
				// Note that in the standard TrackAssociatorByHits, all of the hits in associatedTrackingParticleHits are checked for
				// various things.  I'm not sure what these checks are for but they depend on the UseGrouping and UseSplitting settings.
				// This associator works as though both UseGrouping and UseSplitting were set to true, i.e. just counts the number of
				// hits in the tracker.
				numberOfSimulatedHits=trackingParticleRef->numberOfTrackerHits();
			}

			double purity=static_cast<double>( numberOfSharedHits ) / static_cast<double>( numberOfValidTrackHits );
			double quality;
			if( absoluteNumberOfHits_ ) quality=static_cast<double>( numberOfSharedHits );
			else if( simToRecoDenominator_ == denomsim && numberOfSimulatedHits != 0 ) quality=static_cast<double>( numberOfSharedHits )
					/ static_cast<double>( numberOfSimulatedHits );
			else if( simToRecoDenominator_ == denomreco && numberOfValidTrackHits != 0 ) quality=purity;
			else quality=0;

			if( quality > qualitySimToReco_ && !(threeHitTracksAreSpecial_ && numberOfSimulatedHits == 3 && numberOfSharedHits < 3)
					&& (absoluteNumberOfHits_ || (purity > puritySimToReco_)) )
			{
				returnValue.insert( trackingParticleRef, std::make_pair( edm::RefToBase < TrajectorySeed > (pSeedCollectionHandle_, i), quality ) );
			}
		}
	}
	return returnValue;

	LogTrace("TrackAssociator") << "% of Assoc TPs=" << ((double)returnValue.size())/((double)trackingParticleCollectionHandle->size());
	returnValue.post_insert();
	return returnValue;
}
