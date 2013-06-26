#include "RecoTracker/TkNavigation/plugins/CfgNavigationSchoolESProducer.h"
#include "RecoTracker/TkNavigation/interface/CfgNavigationSchool.h"

NavigationSchoolESProducer::ReturnType CfgNavigationSchoolESProducer::produce(const NavigationSchoolRecord& iRecord){
  using namespace edm::es;

  // get the field
  edm::ESHandle<MagneticField>                field;
  iRecord.getRecord<IdealMagneticFieldRecord>().get(field);

  //get the geometricsearch tracker geometry
  edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;
  iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);

  theNavigationSchool.reset(new CfgNavigationSchool(theNavigationPSet, 
						    geometricSearchTracker.product(), 
						    field.product()) );

  return theNavigationSchool;
}
