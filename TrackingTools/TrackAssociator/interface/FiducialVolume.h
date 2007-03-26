#ifndef TrackAssociator_FiducialVolume_h
#define TrackAssociator_FiducialVolume_h 1

// -*- C++ -*-
//
// Package:    TrackAssociator
// Class:      FiducialVolume
// 
/*

 Description: detector active volume described by a closed cylinder with non-zero thickness.

*/
//
// Original Author:  Dmytro Kovalskyi
// $Id: TrackDetectorAssociator.h,v 1.4 2007/02/19 12:02:41 dmytro Exp $
//
/// The detector active volume is determined in two steps, which are necessary
/// to resolve some ambiguities in determination of inner dimensions of the
/// cylinder. The algorithm assumes that thickness is less than maxR/2, maxZ/2.
/// Z is taken by absolute value.
/// 1) find maxR, maxZ;
/// 2) points with |Z|<maxZ/2 are used to find minR and points with R<maxZ/2
///    are used to find minZ.
// More sophisticated algorithm is needed if any of the following is true:
// - Z thickness > maxZ/2
// - R thickness > maxR/2
// - part of the detector with Z>maxZ/2 has minR less than the region with Z<maxZ/2
// - part of the detector with R>maxR/2 has minZ less than the region with R<maxR/2

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

class FiducialVolume {
 public:
   FiducialVolume(){ reset(); }
   /// check whether the volume is properly defined
   bool isValid() const;
   /// add a point that belongs to the active volume
   void addActivePoint( const GlobalPoint& point );
   /// invalidate the volume
   void reset();
   const double& minR() const { return minR_; }
   const double& maxR() const { return maxR_; }
   const double& minZ() const { return minZ_; }
   const double& maxZ() const { return maxZ_; }
   void determinInnerDimensions();
	
 private:
   double minR_;
   double maxR_;
   double minZ_;
   double maxZ_;
   std::vector<GlobalPoint> activePoints_;
};
#endif
