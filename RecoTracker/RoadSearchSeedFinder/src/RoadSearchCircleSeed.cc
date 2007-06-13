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
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

RoadSearchCircleSeed::RoadSearchCircleSeed(const TrackingRecHit *hit1,
					   const TrackingRecHit *hit2,
					   const TrackingRecHit *hit3,
					   GlobalPoint point1,
					   GlobalPoint point2,
					   GlobalPoint point3) { 

  hits_.push_back(hit1);
  hits_.push_back(hit2);
  hits_.push_back(hit3);

  points_.push_back(point1);
  points_.push_back(point2);
  points_.push_back(point3);

  FastCircle kreis(point1,
		    point2,
		    point3);

  if ( !kreis.isValid() ) {
    // line
    type_ = straightLine;
    center_ = GlobalPoint(0,0,0);
    radius_ = 0;
    impactParameter_ = 0;
  } else {
    type_ = circle;
    radius_          = kreis.rho();
    center_          = GlobalPoint(kreis.x0(),kreis.y0(),0);
    impactParameter_ = calculateImpactParameter(center_,radius_);
  }

}

RoadSearchCircleSeed::RoadSearchCircleSeed(const TrackingRecHit *hit1,
					   const TrackingRecHit *hit2,
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

double RoadSearchCircleSeed::calculateImpactParameter(GlobalPoint center,
						      double radius) {
  //
  // calculate impact parameter to (0,0,0) from center and radius of circle
  //

  double d = std::sqrt( center.x() * center.x() +
			center.y() * center.y() );

  return std::abs(d-radius);
}

std::string RoadSearchCircleSeed::print() const {
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

bool RoadSearchCircleSeed::Compare(const RoadSearchCircleSeed *circle,
				   double centerCut,
				   double radiusCut,
				   unsigned int differentHitsCut) const {
  //
  // compare this circle with the input circle
  // compare: percentage of center difference of center average
  // compare: percentage of radius difference of radius average
  // compare: number of hits which don't overlap between the two circles
  //

  // return value
  bool result = false;

  result = CompareRadius(circle,radiusCut);
  if ( result ) {
    result = CompareCenter(circle,centerCut);
    if ( result ) {
      result = CompareDifferentHits(circle,differentHitsCut);
    }
  }

  return result;

}

bool RoadSearchCircleSeed::CompareCenter(const RoadSearchCircleSeed *circle,
					 double centerCut) const {
  //
  // compare this circle with the input circle
  // compare: percentage of center difference of center average
  //

  // return value
  bool result = false;

  double averageCenter = std::sqrt(((center_.x()+circle->Center().x())/2) *
				   ((center_.x()+circle->Center().x())/2) +
				   ((center_.y()+circle->Center().y())/2) *
				   ((center_.y()+circle->Center().y())/2));
  double differenceCenter = std::sqrt((center_.x()-circle->Center().x()) *
				      (center_.x()-circle->Center().x()) +
				      (center_.y()-circle->Center().y()) *
				      (center_.y()-circle->Center().y()));

  if ( differenceCenter/averageCenter <= centerCut ) {
    result = true;
  }

//   edm::LogVerbatim("OLI") << "center difference: " << differenceCenter
// 			  << "center average: " << averageCenter
// 			  << "center percentage: " << differenceCenter/averageCenter
// 			  << " cut: " << centerCut
// 			  << " result: " << result;

  return result;

}

bool RoadSearchCircleSeed::CompareRadius(const RoadSearchCircleSeed *circle,
					 double radiusCut) const {
  //
  // compare: percentage of center difference of center average
  // compare: percentage of radius difference of radius average
  //

  // return value
  bool result = false;

  double averageRadius = (radius_ + circle->Radius() ) /2;
  double differenceRadius = std::abs(radius_ - circle->Radius());
  
  if ( differenceRadius/averageRadius <= radiusCut ) {
    result = true;
  }

//   edm::LogVerbatim("OLI") << "radius difference: " << differenceRadius
// 			  << " radius average: " << averageRadius
// 			  << " radius percentage: " << differenceRadius/averageRadius
// 			  << " cut: " << radiusCut
// 			  << " result: " << result;

  return result;

}

bool RoadSearchCircleSeed::CompareDifferentHits(const RoadSearchCircleSeed *circle,
						unsigned int differentHitsCut) const {
  //
  // compare this circle with the input circle
  // compare: number of hits which don't overlap between the two circles
  //

  // return value
  bool result = false;

  // assume circles always have 3 hits
  unsigned int counter = 0;
  for ( std::vector<const TrackingRecHit*>::const_iterator hit1 = hits_.begin(),
	  hit1End = hits_.end();
	hit1 != hit1End;
	++hit1 ) {
    bool included = false;
    for ( std::vector<const TrackingRecHit*>::const_iterator hit2 = circle->begin_hits(),
	    hit2End = circle->end_hits();
	  hit2 != hit2End;
	  ++hit2 ) {
      if ( *hit1 == *hit2 ) {
	included = true;
      }
    }
    if ( !included ) {
      ++counter;
    }
  }

  if ( counter <= differentHitsCut ) {
    result = true;
  }

//   edm::LogVerbatim("OLI") << "hits: " << counter 
// 			  << " cut: " << differentHitsCut 
// 			  << " result: " << result;

  return result;

}

