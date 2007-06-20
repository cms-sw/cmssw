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
// $Id: FiducialVolume.h,v 1.1 2007/03/26 05:29:21 dmytro Exp $
//
/// The detector active volume is determined estimated as a non-zero thickness 
/// cylinder with outter dimensions maxZ and maxR. The inner dimensions are
/// found as minimum R and Z for two cases "barrel" (|eta|<1) and 
/// "endcap" (|eta|>1.7) correspondingly

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

class FiducialVolume {
 public:
   FiducialVolume(){ reset(); }
   /// finilize dimension calculations, fixes dimensions in a 
   /// case of missing barrel or endcap
   void determinInnerDimensions();
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
	
 private:
   double minR_;
   double maxR_;
   double minZ_;
   double maxZ_;
};
#endif
