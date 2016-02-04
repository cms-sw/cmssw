#ifndef TkNavigation_SkippingLayerCosmicNavigationSchool_H
#define TkNavigation_SkippingLayerCosmicNavigationSchool_H

#include "RecoTracker/TkNavigation/interface/CosmicNavigationSchool.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include <vector>

//class FakeDetLayer;


/** Concrete navigation school for cosmics in the Tracker
 */

class SkippingLayerCosmicNavigationSchool : public CosmicNavigationSchool {
public:
  SkippingLayerCosmicNavigationSchool(const GeometricSearchTracker* theTracker,
				      const MagneticField* field,
				      const CosmicNavigationSchoolConfiguration conf);

  ~SkippingLayerCosmicNavigationSchool(){cleanMemory();};
};

#endif // SkippingLayerCosmicNavigationSchool_H
