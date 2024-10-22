#ifndef TkDetLayers_SubLayerCrossings_h
#define TkDetLayers_SubLayerCrossings_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#pragma GCC visibility push(hidden)
class SubLayerCrossing {
public:
  SubLayerCrossing() : isValid_(false) {}
  SubLayerCrossing(int sli, int cdi, const GlobalPoint& pos)
      : pos_(pos), subLayerIndex_(sli), closestDetIndex_(cdi), isValid_(true) {}

  bool isValid() { return isValid_; }
  int subLayerIndex() const { return subLayerIndex_; }
  int closestDetIndex() const { return closestDetIndex_; }
  const GlobalPoint& position() const { return pos_; }

private:
  GlobalPoint pos_;
  int subLayerIndex_;
  int closestDetIndex_;
  bool isValid_;
};

class SubLayerCrossings {
public:
  SubLayerCrossings() : isValid_(false) {}
  SubLayerCrossings(const SubLayerCrossing& c, const SubLayerCrossing& o, int ci)
      : closest_(c), other_(o), closestIndex_(ci), isValid_(true) {}

  bool isValid() { return isValid_; }
  const SubLayerCrossing& closest() const { return closest_; }
  const SubLayerCrossing& other() const { return other_; }
  int closestIndex() const { return closestIndex_; }

private:
  SubLayerCrossing closest_;
  SubLayerCrossing other_;
  int closestIndex_;
  bool isValid_;
};

#pragma GCC visibility pop
#endif
