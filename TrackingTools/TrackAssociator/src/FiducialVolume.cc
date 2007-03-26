// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      FiducialVolume
// 
/*

 Description: detector active volume

*/
//
// Original Author:  Dmytro Kovalskyi
// $Id: TrackDetectorAssociator.h,v 1.4 2007/02/19 12:02:41 dmytro Exp $
//
//
#include "TrackingTools/TrackAssociator/interface/FiducialVolume.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

bool FiducialVolume::isValid() const
{
   return minR_>0 && maxR_ >= minR_ && minZ_>0 && maxZ_ >= minZ_;
}

void FiducialVolume::addActivePoint( const GlobalPoint& point )
{
   if ( point.perp() > maxR_ ) maxR_ = point.perp();
   if ( fabs(point.z()) > maxZ_ ) maxZ_ = fabs(point.z());
   activePoints_.push_back(point);
}

void FiducialVolume::reset()
{
   minR_ = -1;
   maxR_ = -1; 
   minZ_ = -1;
   maxZ_ = -1;
   activePoints_.clear();
}

void FiducialVolume::determinInnerDimensions()
{
   if ( maxR_<0 || maxZ_<0 ){
      edm::LogWarning("TrackAssociator") << "outer dimensions are not valid. Cannot compute inner dimensions";
      return;
   }
   for(std::vector<GlobalPoint>::const_iterator point = activePoints_.begin();
       point != activePoints_.end(); ++point)
     {
	if ( point->perp() < maxR_/2 && (minZ_<0 || fabs(point->z()) < minZ_) ) minZ_ = fabs(point->z());
	if ( fabs(point->z()) < maxZ_/2 && (minR_<0 || point->perp() < minR_) ) minR_ = point->perp();
     }
   if ( minZ_ < 0 ) {
      if ( minR_ < 0 ) {
	 throw cms::Exception("FatalError") << "Cannot estimate active volume of the detector";
      } else {
	 minZ_ = maxZ_; // no active endcaps
      }
   } else {
      if ( minR_ < 0 ) {
	 minR_ = maxR_;   // no active barrel
      }
   }
   activePoints_.clear();
}
