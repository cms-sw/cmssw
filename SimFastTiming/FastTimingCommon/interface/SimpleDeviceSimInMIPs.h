#ifndef __SimFastTiming_FastTimingCommon_SimpleDeviceSimInMIPs_h__
#define __SimFastTiming_FastTimingCommon_SimpleDeviceSimInMIPs_h__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

class SimpleDeviceSimInMIPs {
public:
  SimpleDeviceSimInMIPs(const edm::ParameterSet& pset);

  void getEvent(const edm::Event& evt) {}

  void getEventSetup(const edm::EventSetup& evt) {}

  float getChargeForHit(const PSimHit& hit) const { return 1000.f * hit.energyLoss() * MIPPerMeV_; }

private:
  float MIPPerMeV_;
};

#endif
