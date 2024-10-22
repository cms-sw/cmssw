#ifndef _TrimmedVertexFitter_H_
#define _TrimmedVertexFitter_H_

#include "RecoVertex/VertexPrimitives/interface/VertexFitter.h"
#include "RecoVertex/TrimmedKalmanVertexFinder/interface/KalmanTrimmedVertexFinder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
/*
 *  Turn the TrimmedVertexFinder into a VertexFitter.
 */

class TrimmedVertexFitter : public VertexFitter<5> {
public:
  typedef CachingVertex<5>::RefCountedVertexTrack RefCountedVertexTrack;
  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

  TrimmedVertexFitter();
  TrimmedVertexFitter(const edm::ParameterSet& pSet);

  ~TrimmedVertexFitter() override {}

  CachingVertex<5> vertex(const std::vector<reco::TransientTrack>& tracks) const override;

  CachingVertex<5> vertex(const std::vector<RefCountedVertexTrack>& tracks) const override;

  CachingVertex<5> vertex(const std::vector<RefCountedVertexTrack>& tracks, const reco::BeamSpot& spot) const override;

  CachingVertex<5> vertex(const std::vector<reco::TransientTrack>& tracks, const GlobalPoint& linPoint) const override;

  CachingVertex<5> vertex(const std::vector<reco::TransientTrack>& tracks,
                          const GlobalPoint& priorPos,
                          const GlobalError& priorError) const override;

  CachingVertex<5> vertex(const std::vector<RefCountedVertexTrack>& tracks,
                          const GlobalPoint& priorPos,
                          const GlobalError& priorError) const override;

  CachingVertex<5> vertex(const std::vector<reco::TransientTrack>& tracks,
                          const reco::BeamSpot& beamSpot) const override;

  // Clone method
  TrimmedVertexFitter* clone() const override;

  void setPtCut(float cut);
  void setTrackCompatibilityCut(float cut);
  void setVertexFitProbabilityCut(float cut);

private:
  KalmanTrimmedVertexFinder theRector;
  double ptcut;
};

#endif
