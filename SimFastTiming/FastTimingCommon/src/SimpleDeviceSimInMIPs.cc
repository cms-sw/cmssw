#include "SimFastTiming/FastTimingCommon/interface/SimpleDeviceSimInMIPs.h"

SimpleDeviceSimInMIPs::SimpleDeviceSimInMIPs(const edm::ParameterSet& pset)
    : MIPPerMeV_(1.0 / pset.getParameter<double>("meVPerMIP")) {}
