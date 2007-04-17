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

#include <cmath>

#include "RecoTracker/RoadSearchSeedFinder/interface/RoadSearchCircleSeed.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

RoadSearchCircleSeed::RoadSearchCircleSeed(TrackingRecHit *hit1,
					   TrackingRecHit *hit2,
					   TrackingRecHit *hit3,
					   GlobalPoint point1,
					   GlobalPoint point2,
					   GlobalPoint point3) { 

  hits_.push_back(hit1);
  hits_.push_back(hit2);
  hits_.push_back(hit3);

  points_.push_back(point1);
  points_.push_back(point2);
  points_.push_back(point3);

  double points[3][2];

  points[0][0] = point1.x();
  points[0][1] = point1.y();
  points[1][0] = point2.x();
  points[1][1] = point2.y();
  points[2][0] = point3.x();
  points[2][1] = point3.y();

  double array[3][3];

  for ( unsigned int i = 0; i < 3; ++i ) {
    array[i][0] = points[i][0];
    array[i][1] = points[i][1];
    array[i][2] = 1.;
  }
  
  double m11 = determinant(array,3);

  for ( unsigned int i = 0; i < 3; ++i ) {
    array[i][0] = points[i][0]*points[i][0]+points[i][1]*points[i][1];
    array[i][1] = points[i][1];
    array[i][2] = 1.;
  }
  
  double m12 = determinant(array,3);

  for ( unsigned int i = 0; i < 3; ++i ) {
    array[i][0] = points[i][0]*points[i][0]+points[i][1]*points[i][1];
    array[i][1] = points[i][0];
    array[i][2] = 1.;
  }
  
  double m13 = determinant(array,3);

  for ( unsigned int i = 0; i < 3; ++i ) {
    array[i][0] = points[i][0]*points[i][0]+points[i][1]*points[i][1];
    array[i][1] = points[i][0];
    array[i][2] = points[i][1];
  }
  
  double m14 = determinant(array,3);


  if ( std::abs(m11) < 1E-9 ) {
    // line
    type_ = straightLine;
    center_ = GlobalPoint(0,0,0);
    radius_ = 0;
    impactParameter_ = 0;
  } else {
    type_ = circle;
    double x = 0.5 * m12/m11;
    double y = -0.5 * m13/m11;
    radius_  = std::sqrt(x*x+y*y+m14/m11);
    center_ = GlobalPoint(x,y,0);
    impactParameter_ = calculateImpactParameter(center_,radius_);
  }

}

RoadSearchCircleSeed::RoadSearchCircleSeed(TrackingRecHit *hit1,
					   TrackingRecHit *hit2,
					   GlobalPoint point1,
					   GlobalPoint point2) { 
  //
  // straight line constructor
  //


  hits_.push_back(hit1);
  hits_.push_back(hit2);

  points_.push_back(point1);
  points_.push_back(point2);

  type_ = straightLine;
  center_ = GlobalPoint(0,0,0);
  radius_ = 0;
  impactParameter_ = 0;

}

RoadSearchCircleSeed::~RoadSearchCircleSeed() {
}

double RoadSearchCircleSeed::determinant(double array[][3], unsigned int bins) {
  unsigned int i, j, j1, j2;
  double d = 0;
  double temp[3][3];
  for (i = 0; i < 3; ++i ) {
    for (j = 0; j < 3; ++j ) {
      temp[i][j] = 0;
    }
  }

  if (bins == 2)                                // terminate recursion
    {
      d = array[0][0]*array[1][1] - array[1][0]*array[0][1];
    } 
  else 
    {
      d = 0;
      for (j1 = 0; j1 < bins; j1++ )            // do each column
        {
	  for (i = 1; i < bins; i++)            // create minor
            {
	      j2 = 0;
	      for (j = 0; j < bins; j++)
                {
		  if (j == j1) continue;
		  temp[i-1][j2] = array[i][j];
		  j2++;
                }
            }
	  
	  // sum (+/-)cofactor * minor  
	  d = d + std::pow(-1.0, (double)j1)*array[0][j1]*determinant( temp, bins-1 );
        }
    }
  
  return d;

}

double RoadSearchCircleSeed::calculateImpactParameter(GlobalPoint center,
						      double radius) {
  //
  // calculate impact parameter to (0,0,0) from center and radius of circle
  //

  double d = std::sqrt( center.x() * center.x() +
			center.y() * center.y() );

  return std::abs(d-radius);
}

std::string RoadSearchCircleSeed::print() {
  //
  // print function
  //

  std::ostringstream ost;

  if ( type_ == RoadSearchCircleSeed::straightLine ) {
    ost << "Straight Line: number of points: " << points_.size() << "\n";
    unsigned int counter = 0;
    for ( std::vector<GlobalPoint>::const_iterator point = points_.begin();
	  point != points_.end();
	  ++point ) {
      ++counter;
      ost << "    Point " << counter << ": " << point->x() << "," << point->y() << "\n";
    }
  } else {
    ost << "Circle: number of points: " << points_.size() << "\n";
    ost << "    Radius         : " << radius_  << "\n";
    ost << "    ImpactParameter: " << impactParameter_ << "\n";
    ost << "    Center         : " << center_.x() << "," << center_.y() << "\n";
    unsigned int counter = 0;
    for ( std::vector<GlobalPoint>::const_iterator point = points_.begin();
	  point != points_.end();
	  ++point ) {
      ++counter;
      ost << "    Point " << counter << "        : " << point->x() << "," << point->y() << "\n";
    }
  }

  return ost.str(); 
}

std::ostream& operator<<(std::ostream& ost, const RoadSearchCircleSeed & seed) {
  //
  // print operator
  //

  if ( seed.Type() == RoadSearchCircleSeed::straightLine ) {
    ost << "Straight Line: number of points: " << seed.Points().size() << "\n";
    unsigned int counter = 0;
    for ( std::vector<GlobalPoint>::const_iterator point = seed.Points().begin();
	  point != seed.Points().end();
	  ++point ) {
      ++counter;
      ost << "    Point " << counter << ": " << point->x() << "," << point->y() << "\n";
    }
  } else {
    ost << "Circle: number of points: " << seed.Points().size() << "\n";
    ost << "    Radius         : " << seed.Radius()  << "\n";
    ost << "    ImpactParameter: " << seed.ImpactParameter() << "\n";
    ost << "    Center         : " << seed.Center().x() << "," << seed.Center().y() << "\n";
    unsigned int counter = 0;
    for ( std::vector<GlobalPoint>::const_iterator point = seed.Points().begin();
	  point != seed.Points().end();
	  ++point ) {
      ++counter;
      ost << "    Point " << counter << "        : " << point->x() << "," << point->y() << "\n";
    }
  }

  return ost; 
}
