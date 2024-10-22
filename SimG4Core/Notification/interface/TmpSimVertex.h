#ifndef SimG4Core_TmpSimVertex_H
#define SimG4Core_TmpSimVertex_H

#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <cmath>

class TmpSimVertex {
public:
  TmpSimVertex(const math::XYZVectorD& ip, double it, int iv, int typ = 0)
      : ilv_(ip), itime_(it), itrack_(iv), ptype_(typ) {}
  ~TmpSimVertex() = default;
  /// index of the parent (-1 if no parent)
  const math::XYZVectorD& vertexPosition() const { return ilv_; }
  double vertexGlobalTime() const { return itime_; }
  int parentIndex() const { return itrack_; }
  int processType() const { return ptype_; }

private:
  math::XYZVectorD ilv_;
  double itime_;
  int itrack_;
  int ptype_;
};

#endif
