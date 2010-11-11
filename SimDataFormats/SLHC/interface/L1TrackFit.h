
/// ////////////////////////////////////////
/// Written by:                          ///
/// Nicola Pozzobon                      ///
/// UNIPD                                ///
/// 2010, November                       ///
/// Simple class to store L1Track fit    ///
/// ////////////////////////////////////////

#ifndef STACKED_TRACKER_L1_TRACK_FIT_FORMAT_H
#define STACKED_TRACKER_L1_TRACK_FIT_FORMAT_H

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

namespace cmsUpgrades {

class L1TrackFit {

  private:
    double resultCharge;
    double resultRadius;
    GlobalVector resultMomentum;
    GlobalPoint resultVertex;
    GlobalPoint resultAxis;

  public:
    L1TrackFit() {
      resultCharge = 0;
      resultRadius = -999.9;
      resultMomentum = GlobalVector(0.,0.,0.);
      resultVertex = GlobalPoint(0.,0.,0.);
      resultAxis = GlobalPoint(0.,0.,0.);
    }
    
    L1TrackFit(double aCharge, double aRadius, GlobalVector aMomentum, GlobalPoint aVertex, GlobalPoint aAxis) {
      resultCharge = aCharge;
      resultRadius = aRadius;
      resultMomentum = aMomentum;
      resultVertex = aVertex;
      resultAxis = aAxis;
    }    
    
    double getCharge() const {
      return resultCharge;
    }

    double getRadius() const {
      return resultRadius;
    }
    
    GlobalVector getMomentum() const {
      return resultMomentum;
    }

    GlobalPoint getVertex() const {
      return resultVertex;
    }

    GlobalPoint getAxis() const {
      return resultAxis;    
    }
};

}

#endif
