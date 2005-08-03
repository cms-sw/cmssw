#ifndef SimG4Core_Field_H
#define SimG4Core_Field_H

#include "G4MagneticField.hh"

class G4Mag_UsualEqRhs;

class Field : public G4MagneticField
{
public:
    static Field * instance();
    static G4Mag_UsualEqRhs * fieldEquation();
    virtual void GetFieldValue(const double p[3],double b[3]) const;
    void fieldEquation(G4Mag_UsualEqRhs * e);
private:
    Field();
    static Field * theField;
    static G4Mag_UsualEqRhs * theFieldEquation;
};

#endif




