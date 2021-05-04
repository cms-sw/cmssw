#include "SimTransport/PPSProtonTransport/interface/ProtonTransport.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimTransport/PPSProtonTransport/interface/HectorTransport.h"
#include "SimTransport/PPSProtonTransport/interface/OpticalFunctionsTransport.h"
#include "SimTransport/PPSProtonTransport/interface/TotemTransport.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"
#include <CLHEP/Random/RandGauss.h>
#include <CLHEP/Units/GlobalSystemOfUnits.h>
#include <cctype>

ProtonTransport::ProtonTransport(const edm::ParameterSet &iConfig) : instance_(nullptr) {
  std::string transportMethod_ = iConfig.getParameter<std::string>("TransportMethod");
  for (auto &c : transportMethod_)
    c = toupper(c);  // just to on the safe side

  if (transportMethod_ == "HECTOR") {
    instance_ = new HectorTransport(iConfig);
  } else if (transportMethod_ == "TOTEM") {
    instance_ = new TotemTransport(iConfig);
  } else if (transportMethod_ == "OPTICALFUNCTIONS") {
    instance_ = new OpticalFunctionsTransport(iConfig);
  } else {
    throw cms::Exception("ProtonTransport")
        << "Unknow transport method. Is must be either HECTOR, TOTEM or OPTICALFUNCTIONS (case insensitive)";
  }
}
