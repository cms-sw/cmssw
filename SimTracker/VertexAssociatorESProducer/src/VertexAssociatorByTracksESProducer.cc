// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

#include "SimTracker/VertexAssociatorESProducer/src/VertexAssociatorByTracksESProducer.hh"
// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimTracker/VertexAssociation/interface/VertexAssociatorByTracks.h"
#include "SimTracker/Records/interface/VertexAssociatorRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
VertexAssociatorByTracksESProducer::VertexAssociatorByTracksESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this,"VertexAssociatorByTracks");

   //now do what ever other initialization is needed
   conf_=iConfig;
}


VertexAssociatorByTracksESProducer::~VertexAssociatorByTracksESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
VertexAssociatorByTracksESProducer::ReturnType
VertexAssociatorByTracksESProducer::produce(const VertexAssociatorRecord& iRecord)
{
   using namespace edm::es;
   edm::ESHandle<MagneticField> theMF;
   iRecord.getRecord<IdealMagneticFieldRecord>().get(theMF);
   std::auto_ptr<VertexAssociatorBase> pVertexAssociatorBase (new VertexAssociatorByTracks(conf_));
   return pVertexAssociatorBase ;
}

//define this as a plug-in
