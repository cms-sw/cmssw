//
// Package:         RecoTracker/RoadSearchSeedFinder
// Class:           RoadSearchCircleSeed
// 
// Description:     circle from three global points in 2D 
//                  all data members restricted to 2 dimensions
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Mon Jan 22 21:42:35 UTC 2007
//
// $Author: mkirn $
// $Date: 2007/09/07 16:28:52 $
// $Revision: 1.6 $
//

#include <cmath>

#include "RoadSearchCircleSeed.h"
#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

RoadSearchCircleSeed::RoadSearchCircleSeed(const TrackingRecHit *hit1,
					   const TrackingRecHit *hit2,
					   const TrackingRecHit *hit3,
					   GlobalPoint &point1,
					   GlobalPoint &point2,
					   GlobalPoint &point3) { 

  hits_.reserve(3);
  hits_.push_back(hit1);
  hits_.push_back(hit2);
  hits_.push_back(hit3);

  points_.reserve(3);
  points_.push_back(point1);
  points_.push_back(point2);
  points_.push_back(point3);

  FastCircle kreis(point1,
		   point2,
		   point3);

  if ( !kreis.isValid() ) {
    // line
    type_ = straightLine;
    inBarrel_ = true; // Not used for lines
    center_ = GlobalPoint(0,0,0);
    radius_ = 0;
    impactParameter_ = 0;
  } else {
    type_ = circle;
    inBarrel_        = calculateInBarrel();
    radius_          = kreis.rho();
    center_          = GlobalPoint(kreis.x0(),kreis.y0(),0);
    impactParameter_ = calculateImpactParameter(center_,radius_);
  }

}

RoadSearchCircleSeed::RoadSearchCircleSeed(const TrackingRecHit *hit1,
					   const TrackingRecHit *hit2,
					   GlobalPoint &point1,
					   GlobalPoint &point2) { 
  //
  // straight line constructor
  //

  hits_.reserve(2);
  hits_.push_back(hit1);
  hits_.push_back(hit2);

  points_.reserve(2);
  points_.push_back(point1);
  points_.push_back(point2);

  type_ = straightLine;
  inBarrel_ = true; // Not used for lines
  center_ = GlobalPoint(0,0,0);
  radius_ = 0;
  impactParameter_ = 0;

}

RoadSearchCircleSeed::~RoadSearchCircleSeed() {
}

bool RoadSearchCircleSeed::calculateInBarrel() {
  //
  // returns true if all hits are in the barrel,
  // otherwise returns false
  //

  for (std::vector<const TrackingRecHit*>::const_iterator hit = hits_.begin();
       hit != hits_.end(); ++hit) {
    if ((*hit)->geographicalId().subdetId() == StripSubdetector::TEC) {
      return false;
    }
  }
  
  return true;
}

double RoadSearchCircleSeed::calculateImpactParameter(GlobalPoint &center,
						      double radius) {
  //
  // calculate impact parameter to (0,0,0) from center and radius of circle
  //

  double d = std::sqrt( center.x() * center.x() +
			center.y() * center.y() );

  return std::abs(d-radius);
}

double RoadSearchCircleSeed::calculateEta(double theta) const {
  //
  // calculate eta from theta
  //

  return -1.*std::log(std::tan(theta/2.));
}

double RoadSearchCircleSeed::Theta() const {
  //
  // calculate the theta of the seed
  // by taking the average theta of all
  // the lines formed by combinations of
  // hits in the seed
  // 
  // Note:  A faster implementation would
  // calculate in the constructor, save, and
  // return the member here.  This implementation
  // minimizes the memory footprint.
  //

  // Form all the possible lines
  std::vector<LineRZ> lines;
  for (std::vector<GlobalPoint>::const_iterator point1 = points_.begin();
       point1 != points_.end(); ++point1) {
    for (std::vector<GlobalPoint>::const_iterator point2 = point1+1;
	 point2 != points_.end(); ++point2) {
      lines.push_back(LineRZ(*point1, *point2));
    }
  }
  
  double netTheta = 0.;
  for (std::vector<LineRZ>::const_iterator line = lines.begin();
       line != lines.end(); ++line){
    netTheta += line->Theta();
  }
  return netTheta/(double)lines.size();
}

double RoadSearchCircleSeed::Phi0() const {
  //
  // calculate the angle in the x-y plane
  // of the momentum vector at the point of
  // closest approach to (0,0,0)
  //
  // Note:  A faster implementation would
  // calculate in the constructor, save, and
  // return the member here.  This implementation
  // minimizes the memory footprint.
  //

  // Calculate phi as the average phi of all
  // lines formed by combinations of hits if
  // this is a straight line
  if (type_ == straightLine) {
    std::vector<LineXY> lines;
    for (std::vector<GlobalPoint>::const_iterator point1 = points_.begin();
	 point1 != points_.end(); ++point1) {
      for (std::vector<GlobalPoint>::const_iterator point2 = point1+1;
	   point2 != points_.end(); ++point2) {
	lines.push_back(LineXY(*point1,*point2));
      }
    }
    double netPhi = 0.;
    for (std::vector<LineXY>::const_iterator line = lines.begin();
	 line != lines.end(); ++line) {
      netPhi += line->Phi();
    }
    return netPhi/(double)lines.size();
  } // END calculation for linear seeds

  // This calculation is not valid for seeds which do not exit
  // the tracking detector (lines always exit)
  else if (2.*Radius()+ImpactParameter()<110) {
    return 100000.;
  }

  // circular seeds
  else {
    double phi = 100000.;
    double centerPhi = center_.barePhi();

    // Find the first hit in time, which determines the direction of
    // the momentum vector (tangent to the circle at the point of
    // closest approach, clockwise or counter-clockwise).
    // The first hit in time is always the hit with the smallest
    // value r as long as the track exits the tracking detector.
    GlobalPoint firstPoint = points_[0];
    for (unsigned int i=1; i<points_.size(); ++i) {
      if (firstPoint.perp() > points_[i].perp()) {
	firstPoint = points_[i];
      }
    }
    
    // Get the next hit, firstPoint is at the point of
    // closest approach and cannot be used to
    // determine the direction of the initial
    // momentum vector
    if (firstPoint.barePhi() == centerPhi) {
      GlobalPoint nextHit = points_[0];
      for (unsigned int i=1; i<points_.size(); ++i) {
	if (nextHit.perp()  == firstPoint.perp() || 
	    (firstPoint.perp()!= points_[i].perp() &&
	     nextHit.perp() >  points_[i].perp())) {
	  nextHit = points_[i];
	}
      }
      firstPoint = nextHit;
    }
  
    // Find the direction of the momentum vector
    if (firstPoint.barePhi() > centerPhi) {
      // The momentum vector is tangent to
      // the track
      phi = centerPhi + Geom::pi()/2.;
      if (phi>Geom::pi()) {
	phi -= 2.*Geom::pi();
      }
    }
    // Other direction!
    else if (firstPoint.barePhi() < centerPhi) {
      // The momentum vector is tangent to
      // the track
      phi = centerPhi - Geom::pi()/2.;
      if (phi<-1.*Geom::pi()) {
	phi += 2.*Geom::pi();
      }
    }  
    return phi;
  } // END calculation for circular seeds
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
    ost << "    In the barrel  : " << inBarrel_ << "\n";
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
    ost << "    In the barrel  : " << seed.InBarrel() << "\n";
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

//
// constructor
//
LineRZ::LineRZ(GlobalPoint point1, GlobalPoint point2)
{
  theR_ = std::fabs(point1.perp()-point2.perp()); 
  theZ_ = std::fabs(point1.z()-point2.z());
  //If the line is pointing backwards in z
  if ((point1.perp() >= point2.perp() &&
       point1.z() < point2.z()) ||
      (point2.perp() >= point1.perp() &&
       point2.z() < point1.z()))
  {
    theZ_ = -1.*theZ_;
  }
}

//
// destructor
//
LineRZ::~LineRZ()
{ }

//
// constructor
//
LineXY::LineXY(GlobalPoint point1, GlobalPoint point2)
{
  theX_ = std::fabs(point1.x()-point2.x()); 
  theY_ = std::fabs(point1.y()-point2.y());
  //If the line is pointing backwards in x
  if ((point1.perp() >= point2.perp() &&
       point1.x() < point2.x()) ||
      (point2.perp() >= point1.perp() &&
       point2.x() < point1.x()))
  {
    theX_ = -1.*theX_;
  }
  //If the line is pointing backwards in y
  if ((point1.perp() >= point2.perp() &&
       point1.y() < point2.y()) ||
      (point2.perp() >= point1.perp() &&
       point2.y() < point1.y()))
  {
    theY_ = -1.*theY_;
  }
}

//
// destructor
//
LineXY::~LineXY()
{ }
