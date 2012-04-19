
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
    std::vector<double> resultChi2;

  public:
    L1TrackFit() {
      resultCharge = 0;
      resultRadius = -999.9;
      resultMomentum = GlobalVector(0.,0.,0.);
      resultVertex = GlobalPoint(0.,0.,0.);
      resultAxis = GlobalPoint(0.,0.,0.);
      resultChi2.clear();
    }
    
    L1TrackFit(double aCharge, double aRadius, GlobalVector aMomentum, GlobalPoint aVertex, GlobalPoint aAxis, std::vector<double> aChi2) {
      resultCharge = aCharge;
      resultRadius = aRadius;
      resultMomentum = aMomentum;
      resultVertex = aVertex;
      resultAxis = aAxis;
      resultChi2 = aChi2;
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
    
    double getTransChi2() const {
      return resultChi2.at(0);
    }

    double getNormTransChi2() const {
      return resultChi2.at(1);
    }

    double getLongChi2() const {
      return resultChi2.at(2);
    }

    double getNormLongChi2() const {
      return resultChi2.at(3);
    }

};

}

#endif
