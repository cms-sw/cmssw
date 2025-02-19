#include "RecoTracker/TkNavigation/plugins/NavigationSchoolESProducer.h"


//
//
// constructors and destructor
//
NavigationSchoolESProducer::NavigationSchoolESProducer(const edm::ParameterSet& iConfig)
{
  theNavigationPSet = iConfig;
  theNavigationSchoolName = theNavigationPSet.getParameter<std::string>("ComponentName");
  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, theNavigationSchoolName);
  
  //now do what ever other initialization is needed
}


NavigationSchoolESProducer::~NavigationSchoolESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
NavigationSchoolESProducer::ReturnType
NavigationSchoolESProducer::produce(const NavigationSchoolRecord& iRecord)
{
   using namespace edm::es;

   // get the field
   edm::ESHandle<MagneticField>                field;
   iRecord.getRecord<IdealMagneticFieldRecord>().get(field);

   //get the geometricsearch tracker geometry
   edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;
   iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);
   
   theNavigationSchool.reset(NavigationSchoolFactory::get()->create(theNavigationSchoolName,
								    geometricSearchTracker.product(),
								    field.product()));
   return theNavigationSchool ;
}
