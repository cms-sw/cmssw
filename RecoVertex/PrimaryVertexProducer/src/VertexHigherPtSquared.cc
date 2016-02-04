#include "RecoVertex/PrimaryVertexProducer/interface/VertexHigherPtSquared.h"

using namespace reco;

bool 
VertexHigherPtSquared::operator() ( const TransientVertex & v1, 
				    const TransientVertex & v2) const
{
  //  return (sumPtSquared(v1) > sumPtSquared(v2));  V-01-05-02
  std::vector<reco::TransientTrack> tks1 = v1.originalTracks();
  std::vector<reco::TransientTrack> tks2 = v2.originalTracks();
  return (sumPtSquared(tks1) > sumPtSquared(tks2));
}


bool 
VertexHigherPtSquared::operator() ( const Vertex & v1, 
				    const Vertex & v2) const
{
  return (sumPtSquared(v1) > sumPtSquared(v2));
}




double VertexHigherPtSquared::sumPtSquared(const Vertex & v) const 
{
  double sum = 0.;
  double pT;
  for (Vertex::trackRef_iterator it = v.tracks_begin(); it != v.tracks_end(); it++) {
    pT = (**it).pt();
    double epT=(**it).ptError(); 
    pT=pT>epT ? pT-epT : 0;

    sum += pT*pT;
  }
  return sum;
}


double VertexHigherPtSquared::sumPtSquared(const std::vector<reco::TransientTrack> & tks) const 
{
  double sum = 0.;
 for (std::vector<reco::TransientTrack>::const_iterator it = tks.begin(); 
      it != tks.end(); it++) {
   double pT = (it->track()).pt();
   double epT=(it->track()).ptError(); 
   pT=pT>epT ? pT-epT : 0;

   sum += pT*pT;
 }
 return sum;
}
