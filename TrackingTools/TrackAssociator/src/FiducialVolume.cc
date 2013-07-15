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
// $Id: FiducialVolume.cc,v 1.1 2007/03/26 05:29:23 dmytro Exp $
//
//
#include "TrackingTools/TrackAssociator/interface/FiducialVolume.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

bool FiducialVolume::isValid() const
{
   return minR_<1e4 && maxR_ >= minR_ && minZ_<1e4 && maxZ_ >= minZ_;
}

void FiducialVolume::addActivePoint( const GlobalPoint& point )
{
   if ( point.perp() > maxR_ )                              maxR_ = point.perp();
   if ( fabs(point.eta()) < 1 && point.perp() < minR_)      minR_ = point.perp();
   if ( fabs(point.z()) > maxZ_ )                           maxZ_ = fabs(point.z());
   if ( fabs(point.eta()) > 1.7 && fabs(point.z()) < minZ_) minZ_ = fabs(point.z());
}

void FiducialVolume::reset()
{
   minR_ = 1e5;
   maxR_ = -1; 
   minZ_ = 1e5;
   maxZ_ = -1;
}

void FiducialVolume::determinInnerDimensions()
{
   if ( maxR_ > 0 && maxR_ < minR_ ) minR_ = maxR_;
   if ( maxZ_ > 0 && maxZ_ < minZ_ ) minZ_ = maxZ_;
}
