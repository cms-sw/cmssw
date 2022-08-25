#include "MagneticField/Engine/interface/MagneticField.h"
#include "SimG4Core/MagneticField/interface/Field.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

using namespace sim;

Field::Field(const MagneticField *f, double d) : G4MagneticField(), theCMSMagneticField(f), theDelta(d) {
  for (int i = 0; i < 3; ++i) {
    oldx[i] = 1.0e12;
    oldb[i] = 0.0;
  }
}

Field::~Field() {}
