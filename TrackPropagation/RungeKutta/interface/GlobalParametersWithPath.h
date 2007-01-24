#ifndef GlobalParametersWithPath_H
#define GlobalParametersWithPath_H

#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"

class GlobalParametersWithPath {
public:
  GlobalParametersWithPath() : gtp_(), s_(0), valid_(false) {}
  GlobalParametersWithPath( const GlobalTrajectoryParameters& gtp, double s) : 
    gtp_(gtp), s_(s), valid_(true) {}
  GlobalParametersWithPath( const GlobalTrajectoryParameters& gtp, 
			    double s, bool valid) : gtp_(gtp), s_(s), valid_(valid) {}
		 
  const GlobalTrajectoryParameters& parameters() const {return gtp_;}

  double pathLength() const {return s_;}
  double s() const {return pathLength();}

  bool isValid() const {return valid_;}
  operator bool() const {return valid_;}

private:
  GlobalTrajectoryParameters gtp_;
  double s_;
  bool valid_;
};

#endif
