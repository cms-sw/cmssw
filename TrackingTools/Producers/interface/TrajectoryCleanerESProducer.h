// -*- C++ -*-
//
// Package:    TrajectoryCleanerESProducer
// Class:      TrajectoryCleanerESProducer
// 
/**\class TrajectoryCleanerESProducer TrajectoryCleanerESProducer.h TrackingTools/Producers/src/TrajectoryCleanerESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Thu Oct 11 05:20:59 CEST 2007
// $Id: TrajectoryCleanerESProducer.h,v 1.3 2010/04/15 17:56:05 vlimant Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrajectoryCleaning/interface/TrajectoryCleaner.h"

class TrajectoryCleanerESProducer : public edm::ESProducer {
   public:
      TrajectoryCleanerESProducer(const edm::ParameterSet&);
      ~TrajectoryCleanerESProducer();

  typedef boost::shared_ptr<TrajectoryCleaner> ReturnType;

      ReturnType produce(const  TrackingComponentsRecord&);
   private:
  std::string theComponentName;
  std::string theComponentType;
  edm::ParameterSet theConfig;
};
