#ifndef SimTracker_TrackAssociatorProducers_TrackAssociatorByHitsImpl_h
#define SimTracker_TrackAssociatorProducers_TrackAssociatorByHitsImpl_h

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

//reco track
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociatorBaseImpl.h"

class TrackerTopology;

namespace edm {
  class EDProductGetter;
}

class TrackAssociatorByHitsImpl : public reco::TrackToTrackingParticleAssociatorBaseImpl {
  
 public:

  enum SimToRecoDenomType {denomnone,denomsim,denomreco};

  typedef std::vector<std::pair<OmniClusterRef, TrackingParticleRef> > ClusterTPAssociationList;
  typedef std::pair<TrackingParticleRef, TrackPSimHitRef> SimHitTPPair;
  typedef std::vector<SimHitTPPair> SimHitTPAssociationList;


  TrackAssociatorByHitsImpl( edm::EDProductGetter const& productGetter,
                             std::unique_ptr<TrackerHitAssociator> iAssociate,
                             TrackerTopology const* iTopo,
                             SimHitTPAssociationList const* iSimHitsTPAssoc,
                             SimToRecoDenomType iSimToRecoDenominator,
                             double iQuality_SimToReco,
                             double iPurity_SimToReco,
                             double iCut_RecoToSim,
                             bool iUsePixels,
                             bool iUseGrouped,
                             bool iUseSplitting,
                             bool ThreeHitTracksAreSpecial,
                             bool AbsoluteNumberOfHits);


  /* Associate SimTracks to RecoTracks By Hits */
 
  /// Association Reco To Sim with Collections
  virtual
  reco::RecoToSimCollection associateRecoToSim(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&) const override;
  /// Association Sim To Reco with Collections
  virtual
  reco::SimToRecoCollection associateSimToReco(const edm::RefToBaseVector<reco::Track>&,
					       const edm::RefVector<TrackingParticleCollection>&) const override;
  
  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  virtual
  reco::RecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<reco::Track> >& tCH, 
					       const edm::Handle<TrackingParticleCollection>& tPCH) const override {
    return TrackToTrackingParticleAssociatorBaseImpl::associateRecoToSim(tCH,tPCH);
  }
  
  /// compare reco to sim the handle of reco::Track and TrackingParticle collections
  virtual
  reco::SimToRecoCollection associateSimToReco(const edm::Handle<edm::View<reco::Track> >& tCH, 
					       const edm::Handle<TrackingParticleCollection>& tPCH) const override {
    return TrackToTrackingParticleAssociatorBaseImpl::associateSimToReco(tCH,tPCH);
  }  

  //seed
  virtual
  reco::RecoToSimCollectionSeed associateRecoToSim(const edm::Handle<edm::View<TrajectorySeed> >&, 
						   const edm::Handle<TrackingParticleCollection>&) const override;
  
  virtual
  reco::SimToRecoCollectionSeed associateSimToReco(const edm::Handle<edm::View<TrajectorySeed> >&, 
						   const edm::Handle<TrackingParticleCollection>& ) const override;

 private:
  template<typename iter>
  void getMatchedIds(std::vector<SimHitIdpr>&, 
		     std::vector<SimHitIdpr>&, 
		     int&, 
		     iter,
		     iter,
		     TrackerHitAssociator*) const;
  
  int getShared(std::vector<SimHitIdpr>&, 
		std::vector<SimHitIdpr>&,
		TrackingParticle const&) const;

  template<typename iter>
  int getDoubleCount(iter,iter,TrackerHitAssociator*,TrackingParticle const&) const;


  // ----- member data
  edm::EDProductGetter const* productGetter_;
  std::unique_ptr<TrackerHitAssociator> associate;
  TrackerTopology const* tTopo;
  SimHitTPAssociationList const* simHitsTPAssoc;

  SimToRecoDenomType SimToRecoDenominator;
  const double quality_SimToReco;
  const double purity_SimToReco;
  const double cut_RecoToSim;
  const bool UsePixels;
  const bool UseGrouped;
  const bool UseSplitting;
  const bool ThreeHitTracksAreSpecial;
  const bool AbsoluteNumberOfHits;

  const TrackingRecHit* getHitPtr(edm::OwnVector<TrackingRecHit>::const_iterator iter) const {return &*iter;}
  const TrackingRecHit* getHitPtr(trackingRecHit_iterator iter) const {return &**iter;}

  //edm::InputTag _simHitTpMapTag;
};

#endif
