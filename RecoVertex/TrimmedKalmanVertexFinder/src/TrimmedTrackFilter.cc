#include "Utilities/Configuration/interface/Architecture.h"

#include "RecoVertex/TrimmedKalmanVertexFinder/interface/TrimmedTrackFilter.h"


TrimmedTrackFilter::TrimmedTrackFilter()
  : thePtCut(0.) 
{}


bool 
TrimmedTrackFilter::operator() (const TransientTrack& aTk) const 
{
  return aTk.momentumAtVertex().transverse() > thePtCut;
}
