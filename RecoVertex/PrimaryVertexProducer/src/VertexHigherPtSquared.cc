#include "RecoVertex/PrimaryVertexProducer/interface/VertexHigherPtSquared.h"

bool 
VertexHigherPtSquared::operator() ( const TransientVertex & v1, 
				    const TransientVertex & v2) const
{
  std::vector<reco::TransientTrack> tks1 = v1.originalTracks();
  std::vector<reco::TransientTrack> tks2 = v2.originalTracks();
  return (sumPtSquared(tks1) > sumPtSquared(tks2));
}


double 
VertexHigherPtSquared::sumPtSquared(
  const std::vector<reco::TransientTrack> & tks) const 
{
  double sum = 0.;
  for (std::vector<reco::TransientTrack>::const_iterator it = tks.begin(); 
       it != tks.end(); it++) {
    double pT = (*it).impactPointState().globalMomentum().transverse();
    sum += pT*pT;
  }
  return sum;
}
