#ifndef SimG4Core_Field_H
#define SimG4Core_Field_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "G4MagneticField.hh"
#include "SealKernel/Component.h"

class G4Mag_UsualEqRhs;

class Field : public G4MagneticField, public seal::Component
{
    DECLARE_SEAL_COMPONENT;
public:
    Field(seal::Context * c, const edm::ParameterSet & p);
    virtual ~Field();
    static Field * instance();
    static G4Mag_UsualEqRhs * fieldEquation();
    virtual void GetFieldValue(const double p[3],double b[3]) const;
    void fieldEquation(G4Mag_UsualEqRhs * e);
private:
    seal::Context * m_context;
    edm::ParameterSet m_pPhysics; 
    static Field * theField;
    static G4Mag_UsualEqRhs * theFieldEquation;
};

#endif
