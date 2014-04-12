#include <FWCore/Utilities/interface/ESInputTag.h>
#include "RecoTracker/TkNavigation/plugins/CfgNavigationSchoolESProducer.h"
#include "RecoTracker/TkNavigation/interface/CfgNavigationSchool.h"

NavigationSchoolESProducer::ReturnType CfgNavigationSchoolESProducer::produce(const NavigationSchoolRecord& iRecord){
  using namespace edm::es;

  // get the field
  edm::ESHandle<MagneticField>                field;
  std::string mfName = "";
  if (theNavigationPSet.exists("SimpleMagneticField"))
    mfName = theNavigationPSet.getParameter<std::string>("SimpleMagneticField");
  iRecord.getRecord<IdealMagneticFieldRecord>().get(mfName,field);
  //  edm::ESInputTag mfESInputTag(mfName);
  //  iRecord.getRecord<IdealMagneticFieldRecord>().get(mfESInputTag,field);

  //get the geometricsearch tracker geometry
  edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;
  iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);

  theNavigationSchool.reset(new CfgNavigationSchool(theNavigationPSet, 
						    geometricSearchTracker.product(), 
						    field.product()) );

  return theNavigationSchool;
}
