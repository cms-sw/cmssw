#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixLine.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <cmath>
#include <iostream>

// Closest approach of a charged track to the beam line, as computed by TSCBLBuilderNoMaterial:
// tracks produced on the beam line are given by a state far from it (an outer seed), and the
// point of closest approach must come back to the production vertex.

class ConstMagneticField final : public MagneticField {
public:
  ConstMagneticField() { setNominalValue(); }
  GlobalVector inTesla(const GlobalPoint&) const override { return GlobalVector(0, 0, 3.8); }
};

namespace {
  // state reached after turning by alpha along the helix from (vtx, pt, phi, pz)
  GlobalTrajectoryParameters stateAfter(
      GlobalPoint vtx, double pt, double phi, double pz, int q, double alpha, const MagneticField* field) {
    double k = -q * 2.99792458e-3 * field->inTesla(vtx).z() / pt;
    double a = phi + alpha;
    GlobalPoint x(vtx.x() + (std::sin(a) - std::sin(phi)) / k,
                  vtx.y() - (std::cos(a) - std::cos(phi)) / k,
                  vtx.z() + alpha / k * pz / pt);
    return GlobalTrajectoryParameters(x, GlobalVector(pt * std::cos(a), pt * std::sin(a), pz), q, field);
  }
}  // namespace

int main() {
  ConstMagneticField field;
  GlobalPoint bs(0.1, -0.05, 0.3);
  GlobalTrajectoryParameters beam(bs, GlobalVector(1.2e-4, -0.8e-4, 1.), 0, &field);
  int nFail = 0;

  {
    // starting state of cms-sw/cmssw#50121
    GlobalTrajectoryParameters trk(
        GlobalPoint(-53.8587, -64.922, -81.633), GlobalVector(-0.0571817, -0.816178, -0.724604), -1, &field);
    TwoTrackMinimumDistanceHelixLine ttmd;
    bool failed = ttmd.calculate(trk, beam, 0.000001);
    auto pts = ttmd.points();
    double d = (pts.first - pts.second).mag();
    std::cout << "issue 50121: failed " << failed << " pca " << pts.first << " distance " << d << std::endl;
    if (failed || std::abs(d - 0.27477) > 1e-3)
      ++nFail;
  }

  int n = 0;
  for (double pt : {0.3, 0.5, 0.8, 1.2}) {
    for (int q : {-1, 1}) {
      for (double alpha : {0.3, 0.8, 1.4, 2.0, 2.6}) {
        for (int iphi = 0; iphi < 8; ++iphi) {
          GlobalPoint vtx(bs.x() + 1.2e-4f * 2.f, bs.y() - 0.8e-4f * 2.f, bs.z() + 2.f);
          double phi = -3. + 0.8 * iphi, pz = pt * std::sinh(-1.5 + 0.4 * iphi);
          auto trk = stateAfter(vtx, pt, phi, pz, q, -q * alpha, &field);
          // keep starting states within the tracker volume
          if (trk.position().perp() > 120.f || std::abs(trk.position().z()) > 300.f)
            continue;
          TwoTrackMinimumDistanceHelixLine ttmd;
          bool failed = ttmd.calculate(trk, beam, 0.000001);
          double dv = (ttmd.points().first - vtx).mag();
          ++n;
          if (failed || dv > 50e-4) {
            ++nFail;
            std::cout << "FAIL pt " << pt << " q " << q << " turn " << alpha << " start " << trk.position()
                      << " failed " << failed << " distance to vertex " << dv << std::endl;
          }
        }
      }
    }
  }
  std::cout << nFail << " failures out of " << n + 1 << std::endl;
  return nFail == 0 ? 0 : 1;
}
