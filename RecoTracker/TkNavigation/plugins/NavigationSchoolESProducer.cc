#include <FWCore/Utilities/interface/ESInputTag.h>

#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "NavigationSchoolFactory.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"

//
// class decleration
//

class dso_hidden NavigationSchoolESProducer final : public edm::ESProducer {
public:
  NavigationSchoolESProducer(const edm::ParameterSet&);
  ~NavigationSchoolESProducer() override;
  
  typedef std::shared_ptr<NavigationSchool> ReturnType;

  virtual ReturnType produce(const NavigationSchoolRecord&);
 protected:
  // ----------member data ---------------------------
  edm::ParameterSet theNavigationPSet;
  std::string theNavigationSchoolName;
  std::shared_ptr<NavigationSchool> theNavigationSchool ;
};

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
   std::string mfName = "";
   if (theNavigationPSet.exists("SimpleMagneticField"))
     mfName = theNavigationPSet.getParameter<std::string>("SimpleMagneticField");
   iRecord.getRecord<IdealMagneticFieldRecord>().get(mfName,field);
   //   edm::ESInputTag mfESInputTag(mfName);
   //   iRecord.getRecord<IdealMagneticFieldRecord>().get(mfESInputTag,field);

   //get the geometricsearch tracker geometry
   edm::ESHandle<GeometricSearchTracker>         geometricSearchTracker;
   iRecord.getRecord<TrackerRecoGeometryRecord>().get(geometricSearchTracker);
   
   theNavigationSchool.reset(NavigationSchoolFactory::get()->create(theNavigationSchoolName,
								    geometricSearchTracker.product(),
								    field.product()));
   return theNavigationSchool ;
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_EVENTSETUP_MODULE(NavigationSchoolESProducer);
