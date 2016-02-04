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
// $Id: FiducialVolume.h,v 1.5 2011/04/07 08:17:31 innocent Exp $
//
/// The detector active volume is determined estimated as a non-zero thickness 
/// cylinder with outter dimensions maxZ and maxR. The inner dimensions are
/// found as minimum R and Z for two cases "barrel" (|eta|<1) and 
/// "endcap" (|eta|>1.7) correspondingly

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include <vector>

class FiducialVolume {
 public:
   FiducialVolume(double tolerance = 1.0):
     tolerance_(tolerance) { reset(); }
   /// finilize dimension calculations, fixes dimensions in a 
   /// case of missing barrel or endcap
   void determinInnerDimensions();
   /// check whether the volume is properly defined
   bool isValid() const;
   /// add a point that belongs to the active volume
   void addActivePoint( const GlobalPoint& point );
   /// invalidate the volume
   void reset();
   double minR(bool withTolerance = true) const 
     { 
       if (withTolerance && minR_>tolerance_) 
	 return minR_-tolerance_;
       else
	 return minR_;
     }
   double maxR(bool withTolerance = true) const 
     { 
       if (withTolerance) 
	 return maxR_+tolerance_;
       else
	 return maxR_;
     }
   double minZ(bool withTolerance = true) const 
     { 
       if (withTolerance && minZ_>tolerance_)
	 return minZ_-tolerance_; 
       else
	 return minZ_;
     }
   double maxZ(bool withTolerance = true) const 
     { 
       if (withTolerance)
	 return maxZ_+tolerance_;
       else
	 return maxZ_; 
     }
 private:
   double minR_;
   double maxR_;
   double minZ_;
   double maxZ_;
   double tolerance_;
};
#endif
