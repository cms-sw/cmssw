#include "SimTracker/TrackAssociation/interface/QuickTrackAssociatorByHits.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

QuickTrackAssociatorByHits::QuickTrackAssociatorByHits( const edm::ParameterSet& config )
	: pHitAssociator_(NULL), pEventForWhichAssociatorIsValid_(NULL),
	  absoluteNumberOfHits_( config.getParameter<bool>( "AbsoluteNumberOfHits" ) ),
	  qualitySimToReco_( config.getParameter<double>( "Quality_SimToReco" ) ),
	  puritySimToReco_( config.getParameter<double>( "Purity_SimToReco" ) ),
	  cutRecoToSim_( config.getParameter<double>( "Cut_RecoToSim" ) ),
	  threeHitTracksAreSpecial_( config.getParameter<bool> ( "ThreeHitTracksAreSpecial" ) )
{
	//
	// Check whether the denominator when working out the percentage of shared hits should
	// be the number of simulated hits or the number of reconstructed hits.
	//
	std::string denominatorString=config.getParameter<std::string>("SimToRecoDenominator");
	if( denominatorString=="sim" ) simToRecoDenominator_=denomsim;
	else if( denominatorString=="reco" ) simToRecoDenominator_=denomreco;
	else throw cms::Exception( "QuickTrackAssociatorByHits" ) << "SimToRecoDenominator not specified as sim or reco";

	//
	// Set up the parameter set for the hit associator
	//
	hitAssociatorParameters_.addParameter<bool>( "associatePixel", config.getParameter<bool>("associatePixel") );
	hitAssociatorParameters_.addParameter<bool>( "associateStrip", config.getParameter<bool>("associateStrip") );
	// This is the important one, it stops the hit associator searching through the list of sim hits.
	// I only want to use the hit associator methods that work on the hit IDs (i.e. the uint32_t trackId
	// and the EncodedEventId eventId) so I'm not interested in matching that to the PSimHit objects.
	hitAssociatorParameters_.addParameter<bool>("associateRecoTracks",true);

	//
	// Do some checks on whether UseGrouped or UseSplitting have been set. They're not used
	// unlike the standard TrackAssociatorByHits so show a warning.
	//
	bool useGrouped, useSplitting;
	if( config.exists("UseGrouped") ) useGrouped=config.getParameter<bool>("UseGrouped");
	else useGrouped=true;

	if( config.exists("UseSplitting") ) useSplitting=config.getParameter<bool>("UseSplitting");
	else useSplitting=true;

	// This associator works as though both UseGrouped and UseSplitting were set to true, so show a
	// warning if this isn't the case.
	if( !(useGrouped && useSplitting) )
	{
		edm::LogWarning("QuickTrackAssociatorByHits") << "UseGrouped and/or UseSplitting has been set to false, but this associator ignores that setting.";
	}
}

QuickTrackAssociatorByHits::~QuickTrackAssociatorByHits()
{
	delete pHitAssociator_;
}

QuickTrackAssociatorByHits::QuickTrackAssociatorByHits( const QuickTrackAssociatorByHits& otherAssociator )
	: pEventForWhichAssociatorIsValid_(otherAssociator.pEventForWhichAssociatorIsValid_),
	  hitAssociatorParameters_(otherAssociator.hitAssociatorParameters_),
	  absoluteNumberOfHits_(otherAssociator.absoluteNumberOfHits_),
	  qualitySimToReco_(otherAssociator.qualitySimToReco_),
	  puritySimToReco_(otherAssociator.puritySimToReco_),
	  cutRecoToSim_(otherAssociator.cutRecoToSim_),
	  threeHitTracksAreSpecial_(otherAssociator.threeHitTracksAreSpecial_),
	  simToRecoDenominator_(otherAssociator.simToRecoDenominator_)
{
	// No operation other than the initialiser list. That copies everything straight from the other
	// associator, except for pHitAssociator_ which needs a deep copy or both instances will try
	// and free it on deletion.  If it wasn't for pHitAssociator_ the default copy constructor and
	// assignment operator would be sufficient.

	// Actually, need to check the other hit associator isn't null or the pointer dereference would
	// probably cause a segmentation fault.
	if( otherAssociator.pHitAssociator_ ) pHitAssociator_=new TrackerHitAssociator(*otherAssociator.pHitAssociator_);
	else pHitAssociator_=NULL;
}

QuickTrackAssociatorByHits::QuickTrackAssociatorByHits& QuickTrackAssociatorByHits::operator=( const QuickTrackAssociatorByHits& otherAssociator )
{
	// Free up the old pHitAssociator_
	delete pHitAssociator_;

	//
	// pHitAssociator_ needs to be given a deep copy of the object, but everything else can
	// can be shallow copied from the other associator.
	//
	if( otherAssociator.pHitAssociator_ ) pHitAssociator_=new TrackerHitAssociator(*otherAssociator.pHitAssociator_);
	else pHitAssociator_=NULL;
	pEventForWhichAssociatorIsValid_=otherAssociator.pEventForWhichAssociatorIsValid_;
	hitAssociatorParameters_=otherAssociator.hitAssociatorParameters_;
	absoluteNumberOfHits_=otherAssociator.absoluteNumberOfHits_;
	qualitySimToReco_=otherAssociator.qualitySimToReco_;
	puritySimToReco_=otherAssociator.puritySimToReco_;
	cutRecoToSim_=otherAssociator.cutRecoToSim_;
	threeHitTracksAreSpecial_=otherAssociator.threeHitTracksAreSpecial_;
	simToRecoDenominator_=otherAssociator.simToRecoDenominator_;

	return *this;
}

reco::RecoToSimCollection QuickTrackAssociatorByHits::associateRecoToSim(const edm::RefToBaseVector<reco::Track>& trackCollection,
	                                                                                const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection,
	                                                                                const edm::Event* pEvent,
											const edm::EventSetup* pSetup ) const
{
	initialiseHitAssociator( pEvent );

	reco::RecoToSimCollection returnValue;

	for( edm::RefToBaseVector<reco::Track>::const_iterator iTrack=trackCollection.begin(); iTrack!=trackCollection.end(); ++iTrack )
	{
		// The return of this function has first as the index and second as the number of associated hits
		std::pair<edm::Ref<TrackingParticleCollection>,size_t> trackingParticleQualityPair=associateTrack( **iTrack, trackingParticleCollection );
		const edm::Ref<TrackingParticleCollection>& trackingParticleRef=trackingParticleQualityPair.first;
		size_t numberOfSharedHits=trackingParticleQualityPair.second;
		size_t numberOfValidTrackHits=(*iTrack)->found();

		if( numberOfSharedHits==0 ) continue; // No point in continuing if there was no association

		//if electron subtract double counting
		if( abs(trackingParticleRef->pdgId())==11 && (trackingParticleRef->g4Track_end() - trackingParticleRef->g4Track_begin()) > 1 )
		{
			numberOfSharedHits-=getDoubleCount( (*iTrack)->recHitsBegin(), (*iTrack)->recHitsEnd(), *trackingParticleRef );
		}

		double quality;
		if( absoluteNumberOfHits_ ) quality=static_cast<double>( numberOfSharedHits );
		else if( numberOfValidTrackHits != 0 ) quality=(static_cast<double>(numberOfSharedHits) / static_cast<double>(numberOfValidTrackHits) );
		else quality=0;

		if( quality > cutRecoToSim_ && !( threeHitTracksAreSpecial_ && numberOfValidTrackHits==3 && numberOfSharedHits<3 ) )
		{
			returnValue.insert( *iTrack, std::make_pair( trackingParticleRef, quality ));
		}
	}
	return returnValue;
}

reco::SimToRecoCollection QuickTrackAssociatorByHits::associateSimToReco(const edm::RefToBaseVector<reco::Track>& trackCollection,
	                                                                                const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection,
	                                                                                const edm::Event* pEvent,
											const edm::EventSetup* pSetup ) const
{
	initialiseHitAssociator( pEvent );

	reco::SimToRecoCollection returnValue;

	for( edm::RefToBaseVector<reco::Track>::const_iterator iTrack=trackCollection.begin(); iTrack!=trackCollection.end(); ++iTrack )
	{
		// The return of this function has first as an edm:Ref to the associated TrackingParticle, and second as the number of associated hits
		std::pair<edm::Ref<TrackingParticleCollection>,size_t> trackingParticleQualityPair=associateTrack( **iTrack, trackingParticleCollection );
		const edm::Ref<TrackingParticleCollection>& trackingParticleRef=trackingParticleQualityPair.first;
		size_t numberOfSharedHits=trackingParticleQualityPair.second;
		size_t numberOfValidTrackHits=(*iTrack)->found();
		size_t numberOfSimulatedHits=0; // Set a few lines below, but only if required.

		if( numberOfSharedHits==0 ) continue; // No point in continuing if there was no association

		if( simToRecoDenominator_==denomsim || (numberOfSharedHits<3 && threeHitTracksAreSpecial_) ) // the numberOfSimulatedHits is not always required, so can skip counting in some circumstances
		{
			// Note that in the standard TrackAssociatorByHits, all of the hits in associatedTrackingParticleHits are checked for
			// various things.  I'm not sure what these checks are for but they depend on the UseGrouping and UseSplitting settings.
			// This associator works as though both UseGrouping and UseSplitting were set to true, i.e. just counts the number of
			// hits in the tracker.
			numberOfSimulatedHits=trackingParticleRef->trackPSimHit(DetId::Tracker).size();
		}

		double purity=static_cast<double>(numberOfSharedHits)/static_cast<double>(numberOfValidTrackHits);
		double quality;
		if( absoluteNumberOfHits_ ) quality=static_cast<double>(numberOfSharedHits);
		else if( simToRecoDenominator_==denomsim && numberOfSimulatedHits != 0 ) quality=static_cast<double>(numberOfSharedHits)/static_cast<double>(numberOfSimulatedHits);
		else if( simToRecoDenominator_==denomreco && numberOfValidTrackHits != 0 ) quality=purity;
		else quality=0;

		if( quality>qualitySimToReco_ && !( threeHitTracksAreSpecial_ && numberOfSimulatedHits==3 && numberOfSharedHits<3 ) && ( absoluteNumberOfHits_ || (purity>puritySimToReco_) ) )
		{
			returnValue.insert( trackingParticleRef, std::make_pair(*iTrack,quality) );
		}
	}
	return returnValue;

}

std::pair<edm::Ref<TrackingParticleCollection>,size_t> QuickTrackAssociatorByHits::associateTrack( const reco::Track& track, const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection ) const
{
	// I first need to discard any of the invalid hits. I've no idea where they come from but apparently they're completely unphysical.
	// I also need to keep track of which hits I've already associated, so I'll use the same container. I don't need random access but
	// I do need fast removal, so I'll use a linked list.
	std::list<const TrackingRecHit*> unnassociatedHits;
	for( trackingRecHit_iterator iRecHit=track.recHitsBegin(); iRecHit!=track.recHitsEnd(); ++iRecHit )
	{
		if( (*iRecHit)->isValid() ) unnassociatedHits.push_back( &(**iRecHit) );
	}

	// Use these variables to keep track of which TrackingParticle is the best match
	edm::Ref<TrackingParticleCollection> bestAssociatedRef; // This is the array index of the TrackingParticle in trackingParticleCollection
	size_t bestAssociatedNumberOfHits=0;

	// Need to keep track of which TrackingParticles I've checked
	std::vector<const TrackingParticle*> previouslyCheckedParticles;

	// Loop over the unnassociated hits
	std::list<const TrackingRecHit*>::iterator iRecHit=unnassociatedHits.begin();
	while( iRecHit!=unnassociatedHits.end() )
	{
		// See which TrackingParticles are associated to this hit. This is the step that takes a lot of time because it has to loop over
		// and individually check each element in the TrackingParticle collection, which is usually a lot. Therefore minimise the number
		// of times this method has to be called.
		std::vector< edm::Ref<TrackingParticleCollection> > associatedTrackingParticleIndices=associateHitToTrackingParticles( **iRecHit, trackingParticleCollection );

		// Loop over each of these associated TrackingParticles and see how many of the other hits are also associated to these
		// TrackingParticles. This bit shouldn't take as long because the hits are only checked against single TrackingParticles
		// instead of the whole collection.
		for( std::vector< edm::Ref<TrackingParticleCollection> >::const_iterator iParticleRef=associatedTrackingParticleIndices.begin(); iParticleRef!=associatedTrackingParticleIndices.end(); ++iParticleRef )
		{
			// Get a raw reference for ease of use
			const TrackingParticle& particle=**iParticleRef;

			// Make sure I haven't already checked this TrackingParticle already. This could happen if an earlier hit was matched to it.
			// I can't take the earlier matched hit out of unnassociatedHits because it could be associated to more than just this
			// TrackingParticle, and the other TrackingParticle(s) would go unchecked.
			if( std::find( previouslyCheckedParticles.begin(), previouslyCheckedParticles.end(), &particle )!=previouslyCheckedParticles.end() ) continue;
			previouslyCheckedParticles.push_back( &particle );

			// Count how many hits this TrackingParticle can be associated to.
			size_t numberOfAssociatedHits=0;
			for( std::list<const TrackingRecHit*>::const_iterator iRecHitLoop2=unnassociatedHits.begin(); iRecHitLoop2!=unnassociatedHits.end(); ++iRecHitLoop2 )
			{
				if( *iRecHitLoop2 == *iRecHit )
				{
					++numberOfAssociatedHits; // no need to check this hit twice
				}
				else if( trackingParticleContainsHit( particle, **iRecHitLoop2) )
				{
					++numberOfAssociatedHits;
				}
			}

			// See if this TrackingParticle is better associated than the current best match.
			if( numberOfAssociatedHits>bestAssociatedNumberOfHits )
			{
				bestAssociatedRef=*iParticleRef;
				bestAssociatedNumberOfHits=numberOfAssociatedHits;
			}

			// As a short cut I can instantly return if it's impossible for another as yet untested TrackingParticle to have
			// more hits associated. The -1 is because the current hit is still in unnassociatedHits.
			if( unnassociatedHits.size()-1 <= bestAssociatedNumberOfHits ) return std::pair<edm::Ref<TrackingParticleCollection>,size_t>(bestAssociatedRef,bestAssociatedNumberOfHits);
		}

		// I've checked this rec hit against all possible TrackingParticles, so I can take it out of further checks by removing
		// it from the list. Also advance the iterator (erase returns the next iterator) ready for the next loop.
		iRecHit=unnassociatedHits.erase(iRecHit);
	}

	// I would think the shortcut return check in the loop above would have returned by now, but keep this in just in case.
	// I guess if there are no valid hits then the while loop is never entered so that's at least one situation where this
	// is required.
	return std::pair<edm::Ref<TrackingParticleCollection>,size_t>(bestAssociatedRef,bestAssociatedNumberOfHits);

}

std::vector< edm::Ref<TrackingParticleCollection> > QuickTrackAssociatorByHits::associateHitToTrackingParticles( const TrackingRecHit& recHit, const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection ) const
{
	std::vector< edm::Ref<TrackingParticleCollection> > returnValue;

	// This will be filled with the identifiers of any sim tracks that can be
	// matched to the hit. I'll later try and match these identifiers to the
	// sim tracks in the TrackingParticles.
	std::vector< std::pair<uint32_t,EncodedEventId> > simTrackIdentifiers;

	// Figure out which track (or tracks plural if clusters have merged) this hit came from
	pHitAssociator_->associateHitId( recHit, simTrackIdentifiers ); // this call fills simTrackIdentifiers

	if( !simTrackIdentifiers.empty() )
	{
		//
		// Run through the TrackingParticles and see if any of them match the sim track IDs
		//

		// Loop over all the TrackingParticles
		size_t currentIndex=0;
		for( edm::RefVector<TrackingParticleCollection>::const_iterator iTrackingParticle=trackingParticleCollection.begin(); iTrackingParticle!=trackingParticleCollection.end(); ++iTrackingParticle, ++currentIndex )
		{
			const TrackingParticle& trackingParticle=**iTrackingParticle; // Convert to raw reference for ease of use

			// Added this statement because there were differences between my results and the old TrackAssociatorByHits.
			// I think it's down to having this conditional in the old associator.
			if( trackingParticle.trackPSimHit().empty() ) continue;

			// Delegate to another method to see if any of these sim track identifiers are in the TrackingParticle
			if( trackingParticleContainsIdentifiers( trackingParticle, simTrackIdentifiers ) ) returnValue.push_back( *iTrackingParticle );
		}

	}

	return returnValue;
}

bool QuickTrackAssociatorByHits::trackingParticleContainsHit( const TrackingParticle& trackingParticle, const TrackingRecHit& recHit ) const
{
	// Get the parameters that will identify the sim track. The hit associator returns these as a std::pair of the trackId
	// and the eventId.
	std::vector< std::pair<uint32_t,EncodedEventId> > simTrackIdentifiers;
	pHitAssociator_->associateHitId( recHit, simTrackIdentifiers ); // This call fills simTrackIdentifiers

	// Now I have the identifiers I can delegate to the other method.
	return trackingParticleContainsIdentifiers( trackingParticle, simTrackIdentifiers );
}

bool QuickTrackAssociatorByHits::trackingParticleContainsIdentifiers( const TrackingParticle& trackingParticle, const std::vector< std::pair<uint32_t,EncodedEventId> >& identifiers ) const
{
	// Loop over all of the identifiers
	for( std::vector< std::pair<uint32_t,EncodedEventId> >::const_iterator iIdentifier=identifiers.begin(); iIdentifier!=identifiers.end(); ++iIdentifier )
	{
		// Loop over all of the g4 tracks in the tracking particle
		for( std::vector<SimTrack>::const_iterator iSimTrack=trackingParticle.g4Track_begin(); iSimTrack!=trackingParticle.g4Track_end(); ++iSimTrack )
		{
			if( iSimTrack->eventId()==iIdentifier->second && iSimTrack->trackId()==iIdentifier->first )
			{
				return true;
			}
		}
	}

	// If control has made it this far then none of the identifiers were found in
	// any of the g4 tracks, so return false.
	return false;
}

int QuickTrackAssociatorByHits::getDoubleCount( trackingRecHit_iterator startIterator, trackingRecHit_iterator endIterator, const TrackingParticle& associatedTrackingParticle ) const
{
	// This method is largely copied from the standard TrackAssociatorByHits. Once I've tested how much difference
	// it makes I'll go through and comment it properly.

	int doubleCount=0;
	std::vector<SimHitIdpr> SimTrackIdsDC;

	for( trackingRecHit_iterator iHit=startIterator; iHit != endIterator; iHit++ )
	{
		int idcount=0;
		SimTrackIdsDC.clear();
		pHitAssociator_->associateHitId( **iHit, SimTrackIdsDC );

		if( SimTrackIdsDC.size() > 1 )
		{
			for( TrackingParticle::g4t_iterator g4T=associatedTrackingParticle.g4Track_begin(); g4T != associatedTrackingParticle.g4Track_end(); ++g4T )
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


void QuickTrackAssociatorByHits::initialiseHitAssociator( const edm::Event* pEvent ) const
{
	// The intention of this function was to check whether the hit associator is still valid
	// (since in general associateSimToReco and associateRecoToSim are called for the same
	// event). I was doing this by recording the event pointer and checking it hasn't changed
	// but this doesn't appear to work. Until I find a way of uniquely identifying an event
	// I'll just create it anew each time for now.
//	if( pEventForWhichAssociatorIsValid_==pEvent && pEventForWhichAssociatorIsValid_!=NULL ) return; // Already set up so no need to do anything

	// Free up the previous instantiation
	delete pHitAssociator_;

	// Create a new instantiation using the new event
	pHitAssociator_=new TrackerHitAssociator( *pEvent, hitAssociatorParameters_ );
	pEventForWhichAssociatorIsValid_=pEvent;
}

