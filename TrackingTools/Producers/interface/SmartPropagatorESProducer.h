#ifndef TrackingTools_GeomPropagators_SmartPropagatorESProducer_H
#define TrackingTools_GeomPropagators_SmartPropagatorESProducer_H

/** \class SmartPropagatorESProducer
 *  ES producer needed to put the SmartPropagator inside the EventSetup
 *
 *  $Date: 2007/01/17 19:28:40 $
 *  $Revision: 1.1 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "FWCore/Framework/interface/ESProducer.h"

#include "TrackingTools/GeomPropagators/interface/SmartPropagator.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include <boost/shared_ptr.hpp>
  

namespace edm {class ParameterSet;}

class TrackingComponentsRecord;

class  SmartPropagatorESProducer: public edm::ESProducer{

 public:
  
  /// Constructor
  SmartPropagatorESProducer(const edm::ParameterSet &);
  
  /// Destructor
  virtual ~SmartPropagatorESProducer(); 
  
  // Operations
  boost::shared_ptr<Propagator> produce(const TrackingComponentsRecord &);
  
 private:
  boost::shared_ptr<Propagator> thePropagator;
  PropagationDirection thePropagationDirection;
  std::string theTrackerPropagatorName;
  std::string theMuonPropagatorName;
  double theEpsilon;
};

#endif

