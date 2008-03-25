#ifndef SimG4Core_Field_H
#define SimG4Core_Field_H

#include "G4MagneticField.hh"

class MagneticField;
class G4Mag_UsualEqRhs;

namespace sim {
   class Field : public G4MagneticField
   {
      public:
	 Field(const MagneticField * f, double d);
	 virtual ~Field();
	 G4Mag_UsualEqRhs* fieldEquation();
	 virtual void GetFieldValue(const double p[3],double b[3]) const;
	 void fieldEquation(G4Mag_UsualEqRhs* e);
      private:
	 const MagneticField* theCMSMagneticField;
	 G4Mag_UsualEqRhs* theFieldEquation;
	 double theDelta;
   };
}
#endif
