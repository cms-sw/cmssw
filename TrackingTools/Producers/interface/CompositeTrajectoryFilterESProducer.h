#ifndef TrackingTools_CompositeTrajectoryFilter_CompositeTrajectoryFilterESProducer_H
#define TrackingTools_CompositeTrajectoryFilter_CompositeTrajectoryFilterESProducer_H


// -*- C++ -*-
//
// Package:    CompositeTrajectoryFilterESProducer
// Class:      CompositeTrajectoryFilterESProducer
// 
/**\class CompositeTrajectoryFilterESProducer CompositeTrajectoryFilterESProducer.h TrackingTools/TrajectoryFilterESProducer/src/CompositeTrajectoryFilterESProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jean-Roch Vlimant
//         Created:  Fri Sep 28 18:07:52 CEST 2007
// $Id: CompositeTrajectoryFilterESProducer.h,v 1.1 2007/11/15 07:17:07 vlimant Exp $
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

//
// class decleration
//

class CompositeTrajectoryFilterESProducer : public edm::ESProducer {
public:
  CompositeTrajectoryFilterESProducer(const edm::ParameterSet&);
  ~CompositeTrajectoryFilterESProducer();

  typedef std::auto_ptr<TrajectoryFilter> ReturnType;

  ReturnType produce(const TrajectoryFilter::Record &);

private:
  
  std::string componentName;
  std::vector<std::string> filterNames;
};

#endif
