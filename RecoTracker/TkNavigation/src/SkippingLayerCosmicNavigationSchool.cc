#include "RecoTracker/TkNavigation/interface/SkippingLayerCosmicNavigationSchool.h"

SkippingLayerCosmicNavigationSchool::SkippingLayerCosmicNavigationSchool(const GeometricSearchTracker* theInputTracker,
									 const MagneticField* field,
									 const CosmicNavigationSchoolConfiguration conf)
{
  build(theInputTracker, field, conf);
}
