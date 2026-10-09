// The (Bz, Br) r-z lattice of the BrokenLine fit. The fit converts the curvature to pT with the bending field
//   B_bend(r,z) = Bz(r,z) - Br(r,z)*tanLambda*cos(alpha),
// alpha being the angle between the track's transverse direction and the radial direction.
// Both components are normalized to Bz(0,0) and sampled from the MagneticField EventSetup product.
#ifndef RecoTracker_PixelTrackFitting_interface_BLBFieldMap_h
#define RecoTracker_PixelTrackFitting_interface_BLBFieldMap_h

#include <type_traits>

namespace blBFieldMap {
  // Uniform lattice of 5 cm in r and 10 cm in z that covers the tracking volume.
  constexpr int kNR = 24;
  constexpr int kNZ = 57;
  constexpr float kRMax = 115.0f;
  constexpr float kZMax = 280.0f;
  constexpr double kInvDR = double(kNR - 1) / double(kRMax);
  constexpr double kInvDZ = double(kNZ - 1) / (2. * double(kZMax));
  // The normalized Bz and Br at every node, r-major.
  constexpr int kNNodes = kNR * kNZ;
  constexpr inline int nodeIndex(int ir, int iz) { return ir * kNZ + iz; }
  struct Map {
    float bz[kNNodes];
    float br[kNNodes];
  };
  static_assert(std::is_trivially_copyable_v<Map> && std::is_standard_layout_v<Map>);

  // The r and z of lattice index ir, iz; a fractional index gives a point between two nodes.
  constexpr inline double nodeR(double ir) { return double(kRMax) / double(kNR - 1) * ir; }
  constexpr inline double nodeZ(double iz) { return -double(kZMax) + 2.0 * double(kZMax) / double(kNZ - 1) * iz; }

  // r and z clamped to the lattice box.
  constexpr inline double clampR(double r) { return r < 0. ? 0. : (r > double(kRMax) ? double(kRMax) : r); }
  constexpr inline double clampZ(double z) {
    return z < -double(kZMax) ? -double(kZMax) : (z > double(kZMax) ? double(kZMax) : z);
  }

  // A lattice cell: the node (ir, iz) of its lower corner and the fractions tr, tz of a point inside it.
  struct LatticeCell {
    int ir, iz;
    double tr, tz;
  };

  // Bilinear interpolation of one component at the point of cell c.
  constexpr inline double bilinearAt(const float* comp, LatticeCell const& c) {
    const int i00 = nodeIndex(c.ir, c.iz);
    const double b0 = double(comp[i00]) + c.tz * (double(comp[i00 + nodeIndex(0, 1)]) - double(comp[i00]));
    const double b1 = double(comp[i00 + nodeIndex(1, 0)]) +
                      c.tz * (double(comp[i00 + nodeIndex(1, 1)]) - double(comp[i00 + nodeIndex(1, 0)]));
    return b0 + c.tr * (b1 - b0);
  }

  // The lattice cell of (r, z), clamped to the box. The range test before the int conversion also catches NaN.
  constexpr inline LatticeCell cellAt(double r, double z) {
    const double fr = clampR(r) * kInvDR;
    const double fz = (clampZ(z) + double(kZMax)) * kInvDZ;
    const int ir = (fr >= 0. && fr < double(kNR - 2)) ? int(fr) : kNR - 2;
    const int iz = (fz >= 0. && fz < double(kNZ - 2)) ? int(fz) : kNZ - 2;
    return LatticeCell{ir, iz, fr - double(ir), fz - double(iz)};
  }
  // Normalized B_bend(r,z)/Bz(0,0) = Bz - Br*tanLambda*cos(alpha), r/z clamped to the box.
  constexpr inline double bBendAt(const Map& map, double r, double z, double tanLambdaCosAlpha) {
    const LatticeCell c = cellAt(r, z);
    return bilinearAt(map.bz, c) - bilinearAt(map.br, c) * tanLambdaCosAlpha;
  }
  // Normalized B_r(r,z)/Bz(0,0), r/z clamped to the box.
  constexpr inline double brAt(const Map& map, double r, double z) { return bilinearAt(map.br, cellAt(r, z)); }
}  // namespace blBFieldMap
#endif  // RecoTracker_PixelTrackFitting_interface_BLBFieldMap_h
