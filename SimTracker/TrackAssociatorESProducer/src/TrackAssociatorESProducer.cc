// -*- C++ -*-
//
// Package:    TrackAssociatorESProducer
// Class:      TrackAssociatorESProducer
// 
/**\class TrackAssociatorESProducer TrackAssociatorESProducer.h SimTracker/TrackAssociatorESProducer/interface/TrackAssociatorESProducer.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Stefano Magni
//         Created:  Mon Sep 11 17:58:15 CEST 2006
// $Id$
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h" 
//#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h" 
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class decleration
//

class TrackAssociatorESProducer : public edm::ESProducer {
   public:
      TrackAssociatorESProducer(const edm::ParameterSet&);
      ~TrackAssociatorESProducer();

      typedef std::auto_ptr<TrackAssociatorBase> ReturnType;

      ReturnType produce(const TrackAssociatorRecord&);
   private:
      // ----------member data ---------------------------
  edm::ParameterSet conf_;
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
TrackAssociatorESProducer::TrackAssociatorESProducer(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
   conf_=iConfig;
}


TrackAssociatorESProducer::~TrackAssociatorESProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
TrackAssociatorESProducer::ReturnType
TrackAssociatorESProducer::produce(const TrackAssociatorRecord& iRecord)
{
   using namespace edm::es;
   edm::ESHandle<MagneticField> theMF;
   iRecord.getRecord<IdealMagneticFieldRecord>().get(theMF);
   std::auto_ptr<TrackAssociatorBase> pTrackAssociatorBase (new TrackAssociatorByChi2(theMF));
   return pTrackAssociatorBase ;
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(TrackAssociatorESProducer)
