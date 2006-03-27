#include "RecoVertex/PrimaryVertexProducer/interface/VertexHigherPtSquared.h"

bool 
VertexHigherPtSquared::operator() ( const TransientVertex & v1, 
			     const TransientVertex & v2) const
{
  return sumPtSquared(v1.originalTracks()) > sumPtSquared(v2.originalTracks());
}


double 
VertexHigherPtSquared::sumPtSquared(const vector<TransientTrack> & tks) const 
{
  double sum = 0.;
  for (vector<TransientTrack>::const_iterator it = tks.begin(); 
       it != tks.end(); it++) {
    sum += pow((*it).impactPointState().globalMomentum().transverse(), 2);
  }
  return sum;
}
