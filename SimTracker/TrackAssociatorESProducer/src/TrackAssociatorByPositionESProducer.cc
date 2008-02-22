// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByPositionESProducer.hh"
// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackAssociatorByPositionESProducer::TrackAssociatorByPositionESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
  std::string cname = iConfig.getParameter<std::string>("ComponentName");
   setWhatProduced(this,cname);

   //now do what ever other initialization is needed
   conf_=iConfig;
   thePname=iConfig.getParameter<std::string>("propagator");
}


TrackAssociatorByPositionESProducer::~TrackAssociatorByPositionESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
TrackAssociatorByPositionESProducer::ReturnType
TrackAssociatorByPositionESProducer::produce(const TrackAssociatorRecord& iRecord)
{
   using namespace edm::es;
   
   edm::ESHandle<Propagator> theP;
   iRecord.getRecord<TrackingComponentsRecord>().get(thePname,theP);
   
   edm::ESHandle<GlobalTrackingGeometry> theG;
   iRecord.getRecord<GlobalTrackingGeometryRecord>().get(theG);

   std::auto_ptr<TrackAssociatorBase> pTrackAssociatorBase (new TrackAssociatorByPosition(conf_,
											  theG.product(),
											  theP.product()));
   return pTrackAssociatorBase ;
}

//define this as a plug-in
