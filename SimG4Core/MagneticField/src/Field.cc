#include "SimG4Core/MagneticField/interface/Field.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "G4Mag_UsualEqRhs.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

using namespace sim;

G4Mag_UsualEqRhs * Field::fieldEquation() { return theFieldEquation; }

Field::Field(const MagneticField * f, double d) 
  : G4MagneticField(), theCMSMagneticField(f), theDelta(d)
{
  for(int i=0; i<3; ++i) {
    oldx[i] = 1.0e12;
    oldb[i] = 0.0;
  }
}

Field::~Field() {}

void Field::GetFieldValue(const G4double xyz[4], G4double bfield[3]) const 
{ 
  if (std::abs(oldx[0]-xyz[0])>theDelta ||
      std::abs(oldx[1]-xyz[1])>theDelta ||
      std::abs(oldx[2]-xyz[2])>theDelta) 
    {
      static const float lunit = (float)(1.0/CLHEP::cm);
      GlobalPoint ggg((float)(xyz[0])*lunit,(float)(xyz[1])*lunit,(float)(xyz[2])*lunit);
      GlobalVector v = theCMSMagneticField->inTesla(ggg);
      
      static const float btesla = (float)CLHEP::tesla;
      oldb[0] = (G4double)(v.x()*btesla);
      oldb[1] = (G4double)(v.y()*btesla);
      oldb[2] = (G4double)(v.z()*btesla);
      oldx[0] = xyz[0];
      oldx[1] = xyz[1];
      oldx[2] = xyz[2];
    }

  bfield[0] = oldb[0]; 
  bfield[1] = oldb[1]; 
  bfield[2] = oldb[2];
}

void Field::fieldEquation(G4Mag_UsualEqRhs* e) { theFieldEquation = e; }

