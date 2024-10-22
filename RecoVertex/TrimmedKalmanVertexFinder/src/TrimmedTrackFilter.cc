#include "RecoVertex/TrimmedKalmanVertexFinder/interface/TrimmedTrackFilter.h"

using namespace reco;

TrimmedTrackFilter::TrimmedTrackFilter() : thePtCut(0.) {}

bool TrimmedTrackFilter::operator()(const TransientTrack& aTk) const {
  return aTk.initialFreeState().momentum().perp() > thePtCut;
}
