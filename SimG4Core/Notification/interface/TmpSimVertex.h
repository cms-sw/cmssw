#ifndef SimG4Core_TmpSimVertex_H
#define SimG4Core_TmpSimVertex_H

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <cmath>

class TmpSimVertex {
public:
  TmpSimVertex(const math::XYZVectorD& pos, double t, int id, int it, int ptyp)
      : ilv_(pos), itime_(t), idx_{id}, itrack_(it), ptype_(ptyp) {}
  ~TmpSimVertex() = default;
  /// index of the parent (0 if no parent)
  const math::XYZVectorD& vertexPosition() const { return ilv_; }
  double vertexGlobalTime() const { return itime_; }
  int vertexIndex() const { return idx_; }
  int parentIndex() const { return itrack_; }
  int processType() const { return ptype_; }

private:
  math::XYZVectorD ilv_;  // position in cm
  double itime_;          // time in ns
  int idx_;               // vertex index
  int itrack_;            // index of mother track
  int ptype_;             // process type created this vertex
};

#endif
