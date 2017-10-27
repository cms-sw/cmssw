//
// Original Author:  Matt Rudolph
//         Created:  Sat Mar 28 20:13:08 CET 2009
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderWithPropagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"


//
// class decleration
//

class TSCBLBuilderWithPropagatorESProducer : public edm::ESProducer {
   public:
      TSCBLBuilderWithPropagatorESProducer(const edm::ParameterSet&);
      ~TSCBLBuilderWithPropagatorESProducer() override;

      typedef std::shared_ptr<TrajectoryStateClosestToBeamLineBuilder> ReturnType;

      ReturnType produce(const TrackingComponentsRecord&);
   private:
      // ----------member data ---------------------------
      edm::ParameterSet pset_;
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
TSCBLBuilderWithPropagatorESProducer::TSCBLBuilderWithPropagatorESProducer(const edm::ParameterSet& p)
{
   //the following line is needed to tell the framework what
   // data is being produced
  std::string myName = p.getParameter<std::string>("ComponentName");
  pset_ = p;
  setWhatProduced(this,myName);

   //now do what ever other initialization is needed
}


TSCBLBuilderWithPropagatorESProducer::~TSCBLBuilderWithPropagatorESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
TSCBLBuilderWithPropagatorESProducer::ReturnType
TSCBLBuilderWithPropagatorESProducer::produce(const TrackingComponentsRecord& iRecord)
{
   using namespace edm::es;
   std::string propname = pset_.getParameter<std::string>("Propagator");

   edm::ESHandle<Propagator> theProp;
   iRecord.get(propname, theProp);

   const Propagator * pro = theProp.product();

   TSCBLBuilderWithPropagatorESProducer::ReturnType pTSCBLBuilderWithPropagator(new TSCBLBuilderWithPropagator(*pro)) ;


   return pTSCBLBuilderWithPropagator ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(TSCBLBuilderWithPropagatorESProducer);
