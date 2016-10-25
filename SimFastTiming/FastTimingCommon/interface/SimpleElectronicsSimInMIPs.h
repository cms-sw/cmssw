#ifndef __SimFastTiming_FastTimingCommon_SimpleElectronicsSimInMIPs_h__
#define __SimFastTiming_FastTimingCommon_SimpleElectronicsSimInMIPs_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/FTLDigi/interface/FTLDigiCollections.h"
#include "SimFastTiming/FastTimingCommon/interface/FTLDigitizerTypes.h"

namespace ftl = ftl_digitizer;

class SimpleElectronicsSimInMIPs {
 public:
  SimpleElectronicsSimInMIPs(const edm::ParameterSet& pset);
  
  void getEvent(const edm::Event& evt) { }

  void getEventSetup(const edm::EventSetup& evt) { }

  void run(const ftl::FTLSimHitDataAccumulator& input,
	   FTLDigiCollection& output) const;

 private:
  
};

#endif
