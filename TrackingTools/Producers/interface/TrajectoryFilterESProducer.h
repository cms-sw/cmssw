#ifndef TrackingTools_TrajectoryFilter_TrajectoryFilterESProducer_H
#define TrackingTools_TrajectoryFilter_TrajectoryFilterESProducer_H


// -*- C++ -*-
//
// Package:    TrajectoryFilterESProducer
// Class:      TrajectoryFilterESProducer
// 
/**\class TrajectoryFilterESProducer TrajectoryFilterESProducer.h TrackingTools/TrajectoryFilterESProducer/src/TrajectoryFilterESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Sep 28 18:07:52 CEST 2007
// $Id: TrajectoryFilterESProducer.h,v 1.2 2009/04/30 10:14:40 vlimant Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"

// user include files
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TrajectoryFiltering/interface/TrajectoryFilter.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "RecoTracker/Record/interface/CkfComponentsRecord.h"
//
// class decleration
//

class TrajectoryFilterESProducer : public edm::ESProducer {
public:
  TrajectoryFilterESProducer(const edm::ParameterSet&);
  ~TrajectoryFilterESProducer();

  typedef std::auto_ptr<TrajectoryFilter> ReturnType;

  ReturnType produce(const TrajectoryFilter::Record &);
  //  ReturnType produceClusterShapeFilter(const TrajectoryFilter::Record&)
private:
  
  std::string componentName;
  std::string componentType;
  edm::ParameterSet filterPset;
};

#endif
