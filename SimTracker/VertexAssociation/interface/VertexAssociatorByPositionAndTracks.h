#ifndef SimTracker_VertexAssociation_VertexAssociatorByPositionAndTracks_h
#define SimTracker_VertexAssociation_VertexAssociatorByPositionAndTracks_h

#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociatorBaseImpl.h"

/**
 * This class associates reco::Vertices and TrackingVertices by their
 * position (maximum distance in Z should be smaller than absZ and
 * sigmaZ*zError of reco::Vertex), and (optionally) by the fraction of
 * tracks shared by reco::Vertex and TrackingVertex divided by the
 * number of tracks in reco::Vertex. This fraction is always used as
 * the quality in the association, i.e. multiple associations are
 * sorted by it in descending order.
 */
class VertexAssociatorByPositionAndTracks : public reco::VertexToTrackingVertexAssociatorBaseImpl {
public:
  VertexAssociatorByPositionAndTracks(const edm::EDProductGetter *productGetter,
                                      double absZ,
                                      double sigmaZ,
                                      double maxRecoZ,
                                      double absT,
                                      double sigmaT,
                                      double maxRecoT,
                                      double sharedTrackFraction,
                                      const reco::RecoToSimCollection *trackRecoToSimAssociation,
                                      const reco::SimToRecoCollection *trackSimToRecoAssociation);

  VertexAssociatorByPositionAndTracks(const edm::EDProductGetter *productGetter,
                                      double absZ,
                                      double sigmaZ,
                                      double maxRecoZ,
                                      double sharedTrackFraction,
                                      const reco::RecoToSimCollection *trackRecoToSimAssociation,
                                      const reco::SimToRecoCollection *trackSimToRecoAssociation);

  ~VertexAssociatorByPositionAndTracks() override;

  /* Associate TrackingVertex to RecoVertex By Hits */
  reco::VertexRecoToSimCollection associateRecoToSim(const edm::Handle<edm::View<reco::Vertex>> &vCH,
                                                     const edm::Handle<TrackingVertexCollection> &tVCH) const override;

  reco::VertexSimToRecoCollection associateSimToReco(const edm::Handle<edm::View<reco::Vertex>> &vCH,
                                                     const edm::Handle<TrackingVertexCollection> &tVCH) const override;

private:
  // ----- member data
  const edm::EDProductGetter *productGetter_;

  const double absZ_;
  const double sigmaZ_;
  const double maxRecoZ_;
  const double absT_;
  const double sigmaT_;
  const double maxRecoT_;
  const double sharedTrackFraction_;

  const reco::RecoToSimCollection *trackRecoToSimAssociation_;
  const reco::SimToRecoCollection *trackSimToRecoAssociation_;
};

#endif
