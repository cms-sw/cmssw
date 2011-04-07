#ifndef TkDetLayers_SubLayerCrossings_h
#define TkDetLayers_SubLayerCrossings_h

#include "DataFormats/GeometryVector/interface/LocalPoint.h"

#pragma GCC visibility push(hidden)
class SubLayerCrossing {
public:
  SubLayerCrossing(){}
  SubLayerCrossing( int sli, int cdi, const GlobalPoint& pos) : 
    subLayerIndex_(sli), closestDetIndex_(cdi), pos_(pos) {}

  int subLayerIndex()   const {return subLayerIndex_;}
  int closestDetIndex() const {return closestDetIndex_;}
  const GlobalPoint& position() const { return pos_;}

private:
  int subLayerIndex_;
  int closestDetIndex_;
  GlobalPoint pos_;
};

class SubLayerCrossings {
public:
  SubLayerCrossings(): isValid_(false){}
  SubLayerCrossings( const SubLayerCrossing& c, const SubLayerCrossing& o, int ci) :
    isValid_(true),closest_(c), other_(o), closestIndex_(ci) {}

  bool isValid() {return isValid_ ;}
  const SubLayerCrossing& closest() const {return closest_;}
  const SubLayerCrossing& other() const {return other_;}
  int closestIndex() const {return closestIndex_;}

private:
  bool isValid_;
  SubLayerCrossing closest_;
  SubLayerCrossing other_;
  int closestIndex_;

};

#pragma GCC visibility pop
#endif
