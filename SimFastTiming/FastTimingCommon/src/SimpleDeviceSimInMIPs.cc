#include "SimFastTiming/FastTimingCommon/interface/SimpleDeviceSimInMIPs.h"

SimpleDeviceSimInMIPs::SimpleDeviceSimInMIPs(const edm::ParameterSet& pset) : 
  meVPerMIP_( pset.getParameter<double>("meVPerMIP") ) {    
}


