#ifndef VertexAssociatorByTracks_h
#define VertexAssociatorByTracks_h

#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociatorBaseImpl.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

class TrackingParticleSelector;

class VertexAssociatorByTracks : public reco::VertexToTrackingVertexAssociatorBaseImpl {
public:

  VertexAssociatorByTracks(const edm::EDProductGetter *productGetter,
                           double R2SMatchedSimRatio,
                           double R2SMatchedRecoRatio,
                           double S2RMatchedSimRatio,
                           double S2RMatchedRecoRatio,
                           const TrackingParticleSelector *selector,
                           reco::TrackBase::TrackQuality trackQuality,
                           const reco::RecoToSimCollection *trackRecoToSimAssociation,
                           const reco::SimToRecoCollection *trackSimToRecoAssociation);

  ~VertexAssociatorByTracks();

  /* Associate TrackingVertex to RecoVertex By Hits */
  virtual reco::VertexRecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<reco::Vertex> >& vCH, 
                                                             const edm::Handle<TrackingVertexCollection>& tVCH) const;

  virtual reco::VertexSimToRecoCollection associateSimToReco(const edm::Handle<edm::View<reco::Vertex> >& vCH, 
                                                             const edm::Handle<TrackingVertexCollection>& tVCH) const;

private:

  // ----- member data
  const edm::EDProductGetter * productGetter_;

  const double R2SMatchedSimRatio_;
  const double R2SMatchedRecoRatio_;
  const double S2RMatchedSimRatio_;
  const double S2RMatchedRecoRatio_;

  const TrackingParticleSelector *selector_; // Owned by VertexAssociatorByTracksProducer
  const reco::TrackBase::TrackQuality trackQuality_;

  const reco::RecoToSimCollection *trackRecoToSimAssociation_;
  const reco::SimToRecoCollection *trackSimToRecoAssociation_;
};

#endif
