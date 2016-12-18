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
	 virtual void GetFieldValue(const G4double p[4], G4double b[3]) const;
	 void fieldEquation(G4Mag_UsualEqRhs* e);
      private:
	 const MagneticField* theCMSMagneticField;
	 G4Mag_UsualEqRhs* theFieldEquation;
         double theDelta;

         mutable double oldx[3];
         mutable double oldb[3];
   };
}
#endif
