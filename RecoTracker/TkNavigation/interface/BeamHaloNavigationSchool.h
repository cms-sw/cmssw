#ifndef TkNavigation_BeamHaloNavigationSchool_H
#define TkNavigation_BeamHaloNavigationSchool_H

#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include <vector>

/** Concrete navigation school for the Tracker, connecting disks only for traversing tracks : moslty beam halo muon 
 */

class BeamHaloNavigationSchool : public SimpleNavigationSchool {
public:
  
  BeamHaloNavigationSchool(const GeometricSearchTracker* theTracker,
			 const MagneticField* field);
  ~BeamHaloNavigationSchool(){ cleanMemory();}

 protected:
  //addon to SimpleNavigationSchool
  void linkOtherEndLayers( SymmetricLayerFinder& symFinder);
  void addInward(const DetLayer * det, const FDLC& news);
  void addInward(const DetLayer * det, const ForwardDetLayer * newF);
  void establishInverseRelations();
  FDLC reachableFromHorizontal();
};

#endif // BeamHaloNavigationSchool_H
