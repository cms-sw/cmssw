#include "RecoTracker/TkNavigation/plugins/SkippingLayerCosmicNavigationSchoolESProducer.h"
#include "RecoTracker/TkNavigation/interface/SkippingLayerCosmicNavigationSchool.h"

NavigationSchoolESProducer::ReturnType SkippingLayerCosmicNavigationSchoolESProducer::produce(const NavigationSchoolRecord& iRecord){
  using namespace edm::es;

  // get the field
  edm::ESHandle<MagneticField>                field;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(field);

  //get the geometricsearch tracker geometry
  edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;
  iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);

  CosmicNavigationSchool::CosmicNavigationSchoolConfiguration layerConfig(theNavigationPSet);
  theNavigationSchool.reset(new SkippingLayerCosmicNavigationSchool(geometricSearchTracker.product(), field.product(), layerConfig) );

  return theNavigationSchool;
}
