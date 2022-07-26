#ifndef SimG4Core_Field_H
#define SimG4Core_Field_H

#include "G4MagneticField.hh"

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "CLHEP/Units/SystemOfUnits.h"

namespace sim {
  class Field final : public G4MagneticField {
  public:
    Field(const MagneticField *f, double d);
    ~Field() override;
    inline void GetFieldValue(const G4double p[], G4double b[3]) const override;

  private:
    const MagneticField *theCMSMagneticField;
    double theDelta;

    mutable double oldx[3];
    mutable double oldb[3];
  };
};  // namespace sim

void sim::Field::GetFieldValue(const G4double xyz[], G4double bfield[3]) const {
  if (std::abs(oldx[0] - xyz[0]) > theDelta || std::abs(oldx[1] - xyz[1]) > theDelta ||
      std::abs(oldx[2] - xyz[2]) > theDelta) {
    constexpr float lunit = (1.0 / CLHEP::cm);
    GlobalPoint ggg((float)(xyz[0]) * lunit, (float)(xyz[1]) * lunit, (float)(xyz[2]) * lunit);
    GlobalVector v = theCMSMagneticField->inTesla(ggg);

    constexpr float btesla = CLHEP::tesla;
    oldb[0] = (v.x() * btesla);
    oldb[1] = (v.y() * btesla);
    oldb[2] = (v.z() * btesla);
    oldx[0] = xyz[0];
    oldx[1] = xyz[1];
    oldx[2] = xyz[2];
  }

  bfield[0] = oldb[0];
  bfield[1] = oldb[1];
  bfield[2] = oldb[2];
}

#endif
