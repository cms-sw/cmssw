#include "RecoVertex/TertiaryTracksVertexFinder/interface/VertexMass.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

VertexMass::VertexMass() : thePionMass(0.13957) {}

VertexMass::VertexMass(double pionMass) : thePionMass(pionMass) {}

double VertexMass::operator()(const TransientVertex& vtx) const
{

  std::vector<reco::TransientTrack> tracks = vtx.originalTracks();

  double esum=0., pxsum=0., pysum=0., pzsum=0.;

  for(std::vector<reco::TransientTrack>::const_iterator it=tracks.begin();
      it!=tracks.end();it++) {
    reco::TransientTrack track = *it;

    double px = track.impactPointState().globalMomentum().x();
    double py = track.impactPointState().globalMomentum().y();
    double pz = track.impactPointState().globalMomentum().z();

    pxsum += px;
    pysum += py;
    pzsum += pz;
    esum += sqrt(px*px + py*py + pz*pz + thePionMass*thePionMass);
  }
  
  return sqrt(esum*esum - (pxsum*pxsum + pysum*pysum + pzsum*pzsum));
}
