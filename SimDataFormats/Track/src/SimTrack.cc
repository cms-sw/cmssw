#include "SimDataFormats/Track/interface/SimTrack.h"

SimTrack::SimTrack() : ivert(-1), igenpart(-1), crossedBoundary_(false) {}

SimTrack::SimTrack(int ipart, const math::XYZTLorentzVectorD& p)
    : Core(ipart, p), ivert(-1), igenpart(-1), crossedBoundary_(false) {}

SimTrack::SimTrack(int ipart, const math::XYZTLorentzVectorD& p, int iv, int ig)
    : Core(ipart, p), ivert(iv), igenpart(ig), crossedBoundary_(false) {}

SimTrack::SimTrack(int ipart,
                   const math::XYZTLorentzVectorD& p,
                   int iv,
                   int ig,
                   const math::XYZVectorD& tkp,
                   const math::XYZTLorentzVectorD& tkm)
    : Core(ipart, p), ivert(iv), igenpart(ig), tkposition(tkp), tkmomentum(tkm), crossedBoundary_(false) {}

SimTrack::SimTrack(const CoreSimTrack& t, int iv, int ig) : Core(t), ivert(iv), igenpart(ig), crossedBoundary_(false) {}

std::ostream& operator<<(std::ostream& o, const SimTrack& t) {
  return o << (SimTrack::Core)(t) << ", " << t.vertIndex() << ", " << t.genpartIndex();
}
