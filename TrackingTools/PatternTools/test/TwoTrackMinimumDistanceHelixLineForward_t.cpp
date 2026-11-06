#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixLine.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <cmath>
#include <iostream>

// Closest approach of a forward track to the beam line, as computed by TSCBLBuilderNoMaterial:
// starting from its first pixel layers, the point of closest approach must come back to the
// production vertex within a micron.

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
  int n = 0;
  // forward tracks seen from their first pixel layers: the vertex must be found within a micron
  for (double pt : {2., 5., 10.}) {
    for (int q : {-1, 1}) {
      for (double eta : {2.4, 2.7, 3.0}) {
        for (int iphi = 0; iphi < 8; ++iphi) {
          GlobalPoint vtx(bs.x() - 1.2e-4f * 5.f, bs.y() + 0.8e-4f * 5.f, bs.z() - 5.f);
          double phi = -3. + 0.8 * iphi, pz = pt * std::sinh(eta);
          double k = 2.99792458e-3 * 3.8 / pt;
          auto trk = stateAfter(vtx, pt, phi, pz, q, -q * k * 12., &field);  // 12 cm in the transverse plane
          TwoTrackMinimumDistanceHelixLine ttmd;
          bool failed = ttmd.calculate(trk, beam, 0.000001);
          double dv = (ttmd.points().first - vtx).mag();
          ++n;
          if (failed || dv > 1e-4) {
            ++nFail;
            std::cout << "FAIL pt " << pt << " q " << q << " eta " << eta << " start " << trk.position() << " failed "
                      << failed << " distance to vertex " << dv << std::endl;
          }
        }
      }
    }
  }

  std::cout << nFail << " failures out of " << n << std::endl;
  return nFail == 0 ? 0 : 1;
}
