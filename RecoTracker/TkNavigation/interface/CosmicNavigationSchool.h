#ifndef TkNavigation_CosmicNavigationSchool_H
#define TkNavigation_CosmicNavigationSchool_H

//#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "RecoTracker/TkNavigation/interface/SimpleNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
//#include "RecoTracker/TkNavigation/interface/FakeDetLayer.h"

#include <vector>

//class FakeDetLayer;


/** Concrete navigation school for cosmics in the Tracker
 */

class CosmicNavigationSchool : public SimpleNavigationSchool {
public:
  
  CosmicNavigationSchool(const GeometricSearchTracker* theTracker,
			 const MagneticField* field);
 
  ~CosmicNavigationSchool(){};
private:
  //FakeDetLayer* theFakeDetLayer;
  void linkBarrelLayers( SymmetricLayerFinder& symFinder);
  //void linkForwardLayers( SymmetricLayerFinder& symFinder); 
  void establishInverseRelations( SymmetricLayerFinder& symFinder );
  void buildAdditionalBarrelLinks();
  void buildAdditionalForwardLinks(SymmetricLayerFinder& symFinder);
};

#endif // CosmicNavigationSchool_H
