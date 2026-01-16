#ifndef SimTracker_VertexAssociation_calculateVertexSharedTracks_h
#define SimTracker_VertexAssociation_calculateVertexSharedTracks_h

#include "SimDataFormats/Associations/interface/TrackAssociation.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"

struct SharedTracksAndFractions {
  SharedTracksAndFractions(unsigned int nSharedTracks,
                           float sharedTracksFraction,
                           float sharedPt2Fraction,
                           float sharedDzErrFraction)
      : nSharedTracks_(nSharedTracks),
        sharedTracksFraction_(sharedTracksFraction),
        sharedPt2Fraction_(sharedPt2Fraction),
        sharedDzErrFraction_(sharedDzErrFraction) {}

  const unsigned int nSharedTracks_;
  const float sharedTracksFraction_;
  const float sharedPt2Fraction_;
  const float sharedDzErrFraction_;
};

SharedTracksAndFractions calculateVertexSharedTracks(const reco::Vertex &recoV,
                                                     const TrackingVertex &simV,
                                                     const reco::RecoToSimCollection &trackRecoToSimAssociation);

SharedTracksAndFractions calculateVertexSharedTracks(const TrackingVertex &simV,
                                                     const reco::Vertex &recoV,
                                                     const reco::SimToRecoCollection &trackSimToRecoAssociation);

#endif
