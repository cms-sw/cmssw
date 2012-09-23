// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include "SimTracker/TrackAssociatorESProducer/src/TrackAssociatorByChi2ESProducer.hh"
// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "MagneticField/Engine/interface/MagneticField.h" 
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TrackAssociatorByChi2ESProducer::TrackAssociatorByChi2ESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   std::string myName=iConfig.getParameter<std::string>("ComponentName");
   setWhatProduced(this,myName);

   //now do what ever other initialization is needed
   conf_=iConfig;
}


TrackAssociatorByChi2ESProducer::~TrackAssociatorByChi2ESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
TrackAssociatorByChi2ESProducer::ReturnType
TrackAssociatorByChi2ESProducer::produce(const TrackAssociatorRecord& iRecord)
{
   using namespace edm::es;
   edm::ESHandle<MagneticField> theMF;
   iRecord.getRecord<IdealMagneticFieldRecord>().get(theMF);
   std::auto_ptr<TrackAssociatorBase> pTrackAssociatorBase (new TrackAssociatorByChi2(theMF,conf_));
   return pTrackAssociatorBase ;
}

//define this as a plug-in
