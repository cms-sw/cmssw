#ifndef RoadSearchCircleSeed_h
#define RoadSearchCircleSeed_h

//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchCircleSeed
// 
// Description:     circle from three global points in 2D 
//                  all data members restricted to 2 dimensions
//
//                  following http://home.att.net/~srschmitt/circle3pts.html
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Mon Jan 22 21:42:35 UTC 2007
//
// $Author: gutsche $
// $Date: 2007/02/05 19:26:14 $
// $Revision: 1.1 $
//

#include <utility>
#include <vector>
#include <iosfwd>

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"


class RoadSearchCircleSeed 
{
 public:

  // line, first parameter slope, second offset
  typedef std::pair<double,double> line;

  enum type {
    circle,
    straightLine
  };
  
  RoadSearchCircleSeed(TrackingRecHit *hit1,
		       TrackingRecHit *hit2,
		       TrackingRecHit *hit3,
		       GlobalPoint point1,
		       GlobalPoint point2,
		       GlobalPoint point3);
  RoadSearchCircleSeed(TrackingRecHit *hit1,
		       TrackingRecHit *hit2,
		       GlobalPoint point1,
		       GlobalPoint point2);

  ~RoadSearchCircleSeed();

  inline std::vector<GlobalPoint>  Points() const { return points_; }
  inline void AddPoint(GlobalPoint point) { points_.push_back(point); }

  inline std::vector<TrackingRecHit*> Hits() const { return hits_; }
  inline void AddHit(TrackingRecHit* hit) { hits_.push_back(hit); }

  inline GlobalPoint  Center()          const { return center_;}
  inline double       Radius()          const { return radius_;}
  inline double       ImpactParameter() const { return impactParameter_;}
  inline double       Type()            const { return type_; }

  double determinant(double array[][3], unsigned int bins);
  double calculateImpactParameter(GlobalPoint center,
				  double radius);

  std::string print();

 private:

  std::vector<GlobalPoint> points_;

  std::vector<TrackingRecHit*> hits_;

  type        type_;
  GlobalPoint center_;
  double      radius_;
  double      impactParameter_;

};

std::ostream& operator<<(std::ostream& ost, const RoadSearchCircleSeed & seed);

#endif
