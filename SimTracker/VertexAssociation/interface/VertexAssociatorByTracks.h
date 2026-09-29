#ifndef VertexAssociatorByTracks_h
#define VertexAssociatorByTracks_h

#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociatorBaseImpl.h"

class TrackingParticleSelector;

template <typename VertexCollection>
class VertexAssociatorByTracks : public reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection> {
public:
  using VertexType = typename VertexCollection::value_type;
  using SimToRecoCollection = reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>::SimToRecoCollection;
  using RecoToSimCollection = reco::VertexToTrackingVertexAssociatorBaseImpl<VertexCollection>::RecoToSimCollection;

  VertexAssociatorByTracks(const edm::EDProductGetter *productGetter,
                           double R2SMatchedSimRatio,
                           double R2SMatchedRecoRatio,
                           double S2RMatchedSimRatio,
                           double S2RMatchedRecoRatio,
                           const TrackingParticleSelector *selector,
                           reco::TrackBase::TrackQuality trackQuality,
                           const reco::RecoToSimCollection *trackRecoToSimAssociation,
                           const reco::SimToRecoCollection *trackSimToRecoAssociation);

  ~VertexAssociatorByTracks() override = default;

  /* Associate TrackingVertex to RecoVertex By Hits */
  RecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<VertexType>> &vCH,
                                         const edm::Handle<TrackingVertexCollection> &tVCH) const override;

  SimToRecoCollection associateSimToReco(const edm::Handle<edm::View<VertexType>> &vCH,
                                         const edm::Handle<TrackingVertexCollection> &tVCH) const override;

private:
  // ----- member data
  const edm::EDProductGetter *productGetter_;

  const double R2SMatchedSimRatio_;
  const double R2SMatchedRecoRatio_;
  const double S2RMatchedSimRatio_;
  const double S2RMatchedRecoRatio_;

  const TrackingParticleSelector *selector_;  // Owned by VertexAssociatorByTracksProducer
  const reco::TrackBase::TrackQuality trackQuality_;

  const reco::RecoToSimCollection *trackRecoToSimAssociation_;
  const reco::SimToRecoCollection *trackSimToRecoAssociation_;
};

#endif
