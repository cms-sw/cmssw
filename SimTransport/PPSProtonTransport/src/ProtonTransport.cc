#include "SimTransport/PPSProtonTransport/interface/ProtonTransport.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "SimTransport/PPSProtonTransport/interface/HectorTransport.h"
#include "SimTransport/PPSProtonTransport/interface/OpticalFunctionsTransport.h"
#include "SimTransport/PPSProtonTransport/interface/TotemTransport.h"
#include "Utilities/PPS/interface/PPSUnitConversion.h"
#include <cctype>

ProtonTransport::ProtonTransport(const edm::ParameterSet &iConfig, edm::ConsumesCollector iC) {
  std::string transportMethod_ = iConfig.getParameter<std::string>("TransportMethod");
  for (auto &c : transportMethod_)
    c = toupper(c);  // just to on the safe side

  if (transportMethod_ == "HECTOR") {
    instance_ = std::make_unique<HectorTransport>(iConfig, iC);
  } else if (transportMethod_ == "TOTEM") {
    instance_ = std::make_unique<TotemTransport>(iConfig);
  } else if (transportMethod_ == "OPTICALFUNCTIONS") {
    instance_ = std::make_unique<OpticalFunctionsTransport>(iConfig, iC);
  } else {
    throw cms::Exception("ProtonTransport")
        << "Unknow transport method. Is must be either HECTOR, TOTEM or OPTICALFUNCTIONS (case insensitive)";
  }
}
