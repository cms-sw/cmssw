#include "FWCore/Utilities/interface/Exception.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloSimParameters.h"
#include <iostream>

CaloSimParameters::CaloSimParameters(double simHitToPhotoelectrons,
                                     double photoelectronsToAnalog,
                                     double samplingFactor,
                                     double timePhase,
                                     int readoutFrameSize,
                                     int binOfMaximum,
                                     bool doPhotostatistics,
                                     bool syncPhase)
    : simHitToPhotoelectrons_(simHitToPhotoelectrons),
      photoelectronsToAnalog_(photoelectronsToAnalog),
      timePhase_(timePhase),
      readoutFrameSize_(readoutFrameSize),
      binOfMaximum_(binOfMaximum),
      doPhotostatistics_(doPhotostatistics),
      syncPhase_(syncPhase) {}

CaloSimParameters::CaloSimParameters(const edm::ParameterSet &p, bool skipPe2Fc)
    : simHitToPhotoelectrons_(p.getParameter<double>("simHitToPhotoelectrons")),
      photoelectronsToAnalog_(0.),
      timePhase_(p.getParameter<double>("timePhase")),
      readoutFrameSize_(p.getParameter<int>("readoutFrameSize")),
      binOfMaximum_(p.getParameter<int>("binOfMaximum")),
      doPhotostatistics_(p.getParameter<bool>("doPhotoStatistics")),
      syncPhase_(p.getParameter<bool>("syncPhase")) {
  // some subsystems may not want a single number for this
  if (p.existsAs<double>("photoelectronsToAnalog")) {
    photoelectronsToAnalog_ = p.getParameter<double>("photoelectronsToAnalog");
  } else if (p.existsAs<std::vector<double>>("photoelectronsToAnalog")) {
    // just take the first one
    photoelectronsToAnalog_ = p.getParameter<std::vector<double>>("photoelectronsToAnalog").at(0);
  } else if (!skipPe2Fc) {
    throw cms::Exception("CaloSimParameters") << "Cannot find parameter photoelectronsToAnalog";
  }
  // some subsystems may not want this at all
}

std::ostream &operator<<(std::ostream &os, const CaloSimParameters &p) {
  DetId dummy(0);
  os << "CALO SIM PARAMETERS" << std::endl;
  os << p.simHitToPhotoelectrons(dummy) << " pe per SimHit energy " << std::endl;
  os << p.photoelectronsToAnalog() << " Analog signal to be digitized per pe" << std::endl;
  os << " Incident energy / SimHit Energy " << std::endl;
  return os;
}
