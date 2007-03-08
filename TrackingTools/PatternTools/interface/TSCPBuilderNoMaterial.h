#ifndef TSCPBuilderNoMaterial_H
#define TSCPBuilderNoMaterial_H

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToPointBuilder.h"
#include "DataFormats/GeometryVector/interface/GlobalTag.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/GeometryVector/interface/Vector3DBase.h"

/**
 * This class builds a TrajectoryStateClosestToPoint given an original 
 * TrajectoryStateOnSurface or FreeTrajectoryState. This new state is then 
 * defined at the point of closest approach to the reference point.
 */

class TSCPBuilderNoMaterial : 
  public TrajectoryStateClosestToPointBuilder
{
public: 

  virtual ~TSCPBuilderNoMaterial(){}

  virtual TrajectoryStateClosestToPoint operator() 
    (const FTS& originalFTS, const GlobalPoint& referencePoint) const;

  virtual TrajectoryStateClosestToPoint operator() 
    (const TSOS& originalTSOS, const GlobalPoint& referencePoint) const;

private:

  typedef Point3DBase< double, GlobalTag>    GlobalPointDouble;
  typedef Vector3DBase< double, GlobalTag>    GlobalVectorDouble;
  
  FreeTrajectoryState createFTSatTransverseImpactPoint(const FTS& originalFTS, 
      const GlobalPoint& referencePoint) const; 
  
  FreeTrajectoryState createFTSatTransverseImpactPointCharged(const FTS& originalFTS, 
      const GlobalPoint& referencePoint) const; 
  
  FreeTrajectoryState createFTSatTransverseImpactPointNeutral(const FTS& originalFTS, 
      const GlobalPoint& referencePoint) const; 
};
#endif
