///////////////////////////////////////////////////////////////////////////////
// File: EcalSensitiveDetectorBuilder.h
// Description: Builds sensitive detectors for Ecal
///////////////////////////////////////////////////////////////////////////////
#ifndef EcalSensitiveDetectorBuilder_H
#define EcalSensitiveDetectorBuilder_H

#include "Mantis/MantisSensitiveDetector/interface/SensitiveDetectorBuilder.h"
#include "Mantis/MantisSensitiveDetector/interface/SensitiveDetector.h"
#include "Utilities/UI/interface/Verbosity.h"
#include<string>

class EcalSensitiveDetectorBuilder : public SensitiveDetectorBuilder {

public:

  virtual SensitiveDetector*  constructComponent(string);
  virtual string myName();
  EcalSensitiveDetectorBuilder();

private:

  static UserVerbosity cout;

};

#endif
