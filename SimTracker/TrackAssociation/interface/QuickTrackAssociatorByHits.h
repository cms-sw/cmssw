#ifndef QuickTrackAssociatorByHits_h
#define QuickTrackAssociatorByHits_h

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// Forward declarations
class TrackerHitAssociator;

/** @brief TrackAssociator that associates by hits a bit quicker than the normal TrackAssociatorByHits class.
 *
 * NOTE - Doesn't implement the TrajectorySeed or TrackCandidate association methods (from TrackAssociatorBase) so will always
 * return empty associations for those.
 *
 * This track associator (mostly) does the same as TrackAssociatorByHits, but faster. In testing there appears to be some
 * slight differences in the results, but these are minimal. Note that it will only ever return one association, the one with
 * the most shared hits.
 *
 * Configuration parameters:
 *
 * AbsoluteNumberOfHits - bool - if true, Quality_SimToReco and Cut_RecoToSim are the absolute number of shared hits required for
 * association, not the percentage.
 *
 * Quality_SimToReco - double - The minimum amount of shared hits required, as a percentage of either the reconstructed hits or
 * simulated hits (see SimToRecoDenominator), for the track to be considered associated during a call to associateSimToReco. See
 * also AbsoluteNumberOfHits.
 *
 * Purity_SimToReco - double - The minimum amount of shared hits required, as a percentage of the reconstructed hits, for the
 * track to be considered associated during a call to associateSimToReco. Has no effect if AbsoluteNumberOfHits is true.
 *
 * Cut_RecoToSim - double - The minimum amount of shared hits required, as a percentage of the reconstructed hits, for the track
 * to be considered associated during a call to associateRecoToSim. See also AbsoluteNumberOfHits.
 *
 * ThreeHitTracksAreSpecial - bool - If true, tracks with 3 hits must have all their hits associated.
 *
 * SimToRecoDenominator - string - Must be either "sim" or "reco". If "sim" Quality_SimToReco is the percentage of simulated hits
 * that need to be shared. If "reco" then it's the percentage of reconstructed hits (i.e. same as Purity_SimToReco).
 *
 * associatePixel - bool - Passed on to the hit associator.
 *
 * associateStrip - bool - Passed on to the hit associator.
 *
 *
 * Note that the TrackAssociatorByHits parameters UseGrouped and UseSplitting are not used.
 *
 * @author Mark Grimes (mark.grimes@cern.ch)
 * @date 09/Nov/2010
 */
class QuickTrackAssociatorByHits : public TrackAssociatorBase
{
public:
	QuickTrackAssociatorByHits( const edm::ParameterSet& config );
	~QuickTrackAssociatorByHits();
	QuickTrackAssociatorByHits( const QuickTrackAssociatorByHits& otherAssociator );
	QuickTrackAssociatorByHits& operator=( const QuickTrackAssociatorByHits& otherAssociator );

	reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track>& trackCollection,
												 const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection,
												 const edm::Event* pEvent=0,
												 const edm::EventSetup* pSetup=0 ) const;
	reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track>& trackCollection,
												 const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection,
												 const edm::Event* pEvent=0,
												 const edm::EventSetup* pSetup=0 ) const;
private:
	enum SimToRecoDenomType {denomnone,denomsim,denomreco};

	/** @brief Returns the TrackingParticle that has the most associated hits to the given track.
	 *
	 * Return value is a pair, where first is an edm::Ref to the associated TrackingParticle, and second is
	 * the number of associated hits.
	 */
	std::pair<edm::Ref<TrackingParticleCollection>,size_t> associateTrack( const reco::Track& track, const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection ) const;

	/** @brief Returns edm::Refs for all the TrackingParticles that can be associated to the supplied rec hit. */
	std::vector< edm::Ref<TrackingParticleCollection> > associateHitToTrackingParticles( const TrackingRecHit& recHit, const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection ) const;

	/** @brief Returns true if the supplied reconstructed hit can be associated to the supplied TrackingParticle. */
	bool trackingParticleContainsHit( const TrackingParticle& trackingParticle, const TrackingRecHit& recHit ) const;

	/** @brief Returns true if the supplied TrackingParticle has any of the supplied g4 track identifiers. */
	bool trackingParticleContainsIdentifiers( const TrackingParticle& trackingParticle, const std::vector< std::pair<uint32_t,EncodedEventId> >& identifiers ) const;

	/** @brief This method was copied almost verbatim from the standard TrackAssociatorByHits. */
	int getDoubleCount( trackingRecHit_iterator begin, trackingRecHit_iterator end, const TrackingParticle& associatedTrackingParticle ) const;

	//
	// Members. Note that there are custom copy constructor and assignment operators, so if any members are added
	// those methods will need to be updated.
	//
	mutable TrackerHitAssociator* pHitAssociator_;
	const mutable edm::Event* pEventForWhichAssociatorIsValid_;
	void initialiseHitAssociator( const edm::Event* event ) const;

	edm::ParameterSet hitAssociatorParameters_;

	bool absoluteNumberOfHits_;
	double qualitySimToReco_;
	double puritySimToReco_;
	double cutRecoToSim_;
	bool threeHitTracksAreSpecial_;
	SimToRecoDenomType simToRecoDenominator_;

}; // end of the QuickTrackAssociatorByHits class

#endif // end of ifndef QuickTrackAssociatorByHits_h
