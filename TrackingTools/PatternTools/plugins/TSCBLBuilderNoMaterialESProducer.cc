//
// Original Author:  Boris Mangano
//         Created:  Sat Mar 28 20:13:08 CET 2009
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"


//
// class decleration
//

class TSCBLBuilderNoMaterialESProducer : public edm::ESProducer {
   public:
      TSCBLBuilderNoMaterialESProducer(const edm::ParameterSet&);
      ~TSCBLBuilderNoMaterialESProducer();

      typedef boost::shared_ptr<TrajectoryStateClosestToBeamLineBuilder> ReturnType;

      ReturnType produce(const TrackingComponentsRecord&);
   private:
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TSCBLBuilderNoMaterialESProducer::TSCBLBuilderNoMaterialESProducer(const edm::ParameterSet& p)
{
   //the following line is needed to tell the framework what
   // data is being produced
  std::string myName = p.getParameter<std::string>("ComponentName");
  setWhatProduced(this,myName);

   //now do what ever other initialization is needed
}


TSCBLBuilderNoMaterialESProducer::~TSCBLBuilderNoMaterialESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
TSCBLBuilderNoMaterialESProducer::ReturnType
TSCBLBuilderNoMaterialESProducer::produce(const TrackingComponentsRecord& iRecord)
{
   using namespace edm::es;
   TSCBLBuilderNoMaterialESProducer::ReturnType pTSCBLBuilderNoMaterial(new TSCBLBuilderNoMaterial()) ;


   return pTSCBLBuilderNoMaterial ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(TSCBLBuilderNoMaterialESProducer);
