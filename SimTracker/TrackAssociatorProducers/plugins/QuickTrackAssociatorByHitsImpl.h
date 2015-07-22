#ifndef QuickTrackAssociatorByHitsImpl_h
#define QuickTrackAssociatorByHitsImpl_h

#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"

#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociationList.h"

// Forward declarations
class TrackerHitAssociator;

namespace edm {
  class EDProductGetter;
}

/** @brief TrackToTrackingParticleAssociator that associates by hits a bit quicker than the normal TrackAssociatorByHitsImpl class.
 *
 * NOTE - Doesn't implement the TrackCandidate association methods (from TrackAssociatorBase) so will always
 * return empty associations for those.
 *
 * This track associator (mostly) does the same as TrackAssociatorByHitsImpl, but faster. I've tested it a fair bit and can't find
 * any differences between the results of this and the standard TrackAssociatorByHitsImpl.
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
 * requireStoredHits - bool - Whether or not to insist all TrackingParticles have at least one PSimHit. The PSimHits are not required
 * for the association, but the old TrackAssociatorByHitsImpl still had this requirement. Storing PSimHits in the TrackingParticle is now
 * optional (see TrackingTruthAccumulator which replaces TrackingTruthProducer). Having requireStoredHits set to true will mean no
 * TrackingParticles will be associated if you have chosen not to store the hits. The flag is only kept in order to retain the old
 * behaviour which can give very slightly different results.
 *
 * Note that the TrackAssociatorByHitsImpl parameters UseGrouped and UseSplitting are not used.
 *
 * @author Mark Grimes (mark.grimes@cern.ch)
 * @date 09/Nov/2010
 * Significant changes to remove any differences to the standard TrackAssociatorByHitsImpl results 07/Jul/2011.
 * Association for TrajectorySeeds added by Giuseppe Cerati sometime between 2011 and 2013.
 * Functionality to associate using pre calculated cluster to TrackingParticle maps added by Subir Sarker sometime in 2013.
 * Overhauled to remove mutables to make it thread safe by Mark Grimes 01/May/2014.
 */
class QuickTrackAssociatorByHitsImpl : public reco::TrackToTrackingParticleAssociatorBaseImpl
{
public:
  enum SimToRecoDenomType {denomnone,denomsim,denomreco};

  QuickTrackAssociatorByHitsImpl(edm::EDProductGetter const& productGetter,
                                 std::unique_ptr<const TrackerHitAssociator> hitAssoc,
                                 const ClusterTPAssociationList *clusterToTPMap,
                                 bool absoluteNumberOfHits,
                                 double qualitySimToReco,
                                 double puritySimToReco,
                                 double cutRecoToSim,
                                 bool threeHitTracksAreSpecial,
                                 SimToRecoDenomType simToRecoDenominator);
  
  virtual
    reco::RecoToSimCollection associateRecoToSim( const edm::Handle<edm::View<reco::Track> >& trackCollectionHandle,
                                                  const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const override;
  virtual
    reco::SimToRecoCollection associateSimToReco( const edm::Handle<edm::View<reco::Track> >& trackCollectionHandle,
                                                  const edm::Handle<TrackingParticleCollection>& trackingParticleCollectionHandle) const override;
  virtual
    reco::RecoToSimCollection associateRecoToSim( const edm::RefToBaseVector<reco::Track>& trackCollection,
                                                  const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection) const override;
  
  virtual
    reco::SimToRecoCollection associateSimToReco( const edm::RefToBaseVector<reco::Track>& trackCollection,
                                                  const edm::RefVector<TrackingParticleCollection>& trackingParticleCollection) const override;
  
  //seed
  virtual
    reco::RecoToSimCollectionSeed associateRecoToSim(const edm::Handle<edm::View<TrajectorySeed> >&,
                                                     const edm::Handle<TrackingParticleCollection>&) const override;
  
  virtual
    reco::SimToRecoCollectionSeed associateSimToReco(const edm::Handle<edm::View<TrajectorySeed> >&,
                                                     const edm::Handle<TrackingParticleCollection>&) const override;
  
  
 private:
  typedef std::pair<uint32_t,EncodedEventId> SimTrackIdentifiers; ///< @brief This is enough information to uniquely identify a sim track
  
  // - added by S. Sarkar
  static bool tpIntPairGreater(std::pair<edm::Ref<TrackingParticleCollection>,size_t> i, std::pair<edm::Ref<TrackingParticleCollection>,size_t> j) { return (i.first.key()>j.first.key()); }
  
  /** @brief The method that does the work for both overloads of associateRecoToSim.
   *
   * Parts that actually rely on the type of the collections are delegated out to overloaded functions
   * in the unnamed namespace of the .cc file. Parts that rely on the type of T_hitOrClusterAssociator
   * are delegated out to overloaded methods.
   */
  template<class T_TrackCollection, class T_TrackingParticleCollection, class T_hitOrClusterAssociator>
    reco::RecoToSimCollection associateRecoToSimImplementation( T_TrackCollection trackCollection, T_TrackingParticleCollection trackingParticleCollection, T_hitOrClusterAssociator hitOrClusterAssociator ) const;
  
  /** @brief The method that does the work for both overloads of associateSimToReco.
   *
   * Parts that actually rely on the type of the collections are delegated out to overloaded functions
   * in the unnamed namespace of the .cc file. Parts that rely on the type of T_hitOrClusterAssociator
   * are delegated out to overloaded methods.
   */
  template<class T_TrackCollection, class T_TrackingParticleCollection, class T_hitOrClusterAssociator>
    reco::SimToRecoCollection associateSimToRecoImplementation( T_TrackCollection trackCollection, T_TrackingParticleCollection trackingParticleCollection, T_hitOrClusterAssociator hitOrClusterAssociator ) const;
  
  
  /** @brief Returns the TrackingParticle that has the most associated hits to the given track.
   *
   * Return value is a vector of pairs, where first is an edm::Ref to the associated TrackingParticle, and second is
   * the number of associated hits.
   */
  template<typename T_TPCollection,typename iter> std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > associateTrack( const TrackerHitAssociator& hitAssociator, T_TPCollection trackingParticles, iter begin, iter end ) const;
  /** @brief Returns the TrackingParticle that has the most associated hits to the given track.
   *
   * See the notes for the other overload for the return type.
   *
   * Note that the trackingParticles parameter is not actually required since all the information is in clusterToTPMap,
   * but the method signature has to match the other overload because it is called from a templated method.
   */
  template<typename T_TPCollection,typename iter> std::vector< std::pair<edm::Ref<TrackingParticleCollection>,size_t> > associateTrack( const ClusterTPAssociationList& clusterToTPMap, T_TPCollection trackingParticles, iter begin, iter end ) const;
  
  
  /** @brief Returns true if the supplied TrackingParticle has the supplied g4 track identifiers. */
  bool trackingParticleContainsIdentifier( const TrackingParticle* pTrackingParticle, const SimTrackIdentifiers& identifier ) const;
  
  /** @brief This method was copied almost verbatim from the standard TrackAssociatorByHits.
   *
   * Modified 01/May/2014 to take the TrackerHitAssociator as a parameter rather than using a member.
   */
  template<typename iter> int getDoubleCount( const TrackerHitAssociator& hitAssociator, iter begin, iter end, TrackingParticleRef associatedTrackingParticle ) const;
  /** @brief Overload for when using cluster to TrackingParticle association list.
   */
  template<typename iter> int getDoubleCount( const ClusterTPAssociationList& clusterToTPList, iter begin, iter end, TrackingParticleRef associatedTrackingParticle ) const;
  
  /** @brief Returns a vector of pairs where first is a SimTrackIdentifiers (see typedef above) and second is the number of hits that came from that sim track.
   *
   * This is used so that the TrackingParticle collection only has to be looped over once to search for each sim track, rather than once per hit.
   * E.g. If all the hits in the reco track come from the same sim track, then there will only be one entry with second as the number of hits in
   * the track.
   */
  template<typename iter> std::vector< std::pair<SimTrackIdentifiers,size_t> > getAllSimTrackIdentifiers( const TrackerHitAssociator& hitAssociator, iter begin, iter end ) const;
  
  // Added by S. Sarkar
  template<typename iter> std::vector< OmniClusterRef> getMatchedClusters( iter begin, iter end ) const;
  
  const TrackingRecHit* getHitFromIter(trackingRecHit_iterator iter) const {
    return &(**iter);
  }
  
  const TrackingRecHit* getHitFromIter(TrackingRecHitCollection::const_iterator iter) const {
    return &(*iter);
  }
  
  /** @brief creates either a ClusterTPAssociationList OR a TrackerHitAssociator and stores it in the provided unique_ptr. The other will be null.
   *
   * A decision is made whether to create a ClusterTPAssociationList or a TrackerHitAssociator depending on how this
   * track associator was configured. If the ClusterTPAssociationList couldn't be fetched from the event then it
   * falls back to creating a TrackerHitAssociator.
   *
   * Only one type will be created, never both. The other unique_ptr reference will be null so check for that
   * and decide which to use.
   *
   * N.B. The value of useClusterTPAssociation_ should not be used to decide which of the two pointers to use. If
   * the cluster to TrackingParticle couldn't be retrieved from the event then pClusterToTPMap will be null but
   * useClusterTPAssociation_ is no longer changed to false.
   */
  //void prepareEitherHitAssociatorOrClusterToTPMap( const edm::Event* pEvent, std::unique_ptr<ClusterTPAssociationList>& pClusterToTPMap, std::unique_ptr<TrackerHitAssociator>& pHitAssociator ) const;

  edm::EDProductGetter const* productGetter_;
  std::unique_ptr<const TrackerHitAssociator> hitAssociator_;
  const ClusterTPAssociationList *clusterToTPMap_;
  
  double qualitySimToReco_;
  double puritySimToReco_;
  double cutRecoToSim_;
  SimToRecoDenomType simToRecoDenominator_;
  bool threeHitTracksAreSpecial_;
  bool absoluteNumberOfHits_;
  
  // Added by S. Sarkar
  //bool useClusterTPAssociation_;
}; // end of the QuickTrackAssociatorByHitsImpl class

#endif // end of ifndef QuickTrackAssociatorByHitsImpl_h
