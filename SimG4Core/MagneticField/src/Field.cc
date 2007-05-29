#include "SimG4Core/MagneticField/interface/Field.h"
#include "MagneticField/Engine/interface/MagneticField.h"

//#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "G4Mag_UsualEqRhs.hh"

#include "CLHEP/Units/SystemOfUnits.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"

using namespace sim;

G4Mag_UsualEqRhs * Field::fieldEquation() { return theFieldEquation; }

Field::Field(const MagneticField * f,const edm::ParameterSet & p) 
    : G4MagneticField(), theCMSMagneticField(f),m_pField(p)
{
}

Field::~Field() {}

void Field::GetFieldValue(const double xyz[3],double bfield[3]) const 
{ 

    // double protection (in principal, just isnan() should do the job...)
    //
    if ( !(xyz[0]==xyz[0]) || !(xyz[1]==xyz[1]) || !(xyz[2]==xyz[2]) ||
         isnan(xyz[0]) != 0 || isnan(xyz[1]) != 0 || isnan(xyz[2]) != 0 )
    {
       throw SimG4Exception( "SimG4CoreMagneticField: Corrupted Event - NaN detected (position)" ) ;
    }
    
    double delta = 1.;

    static float oldx[3] = {1.0e12,1.0e12,1.0e12};
    static double b[3];

    if (delta>0. &&
	fabs(oldx[0]-xyz[0])<delta &&
	fabs(oldx[1]-xyz[1])<delta &&
	fabs(oldx[2]-xyz[2])<delta) 
    {
	// old b good enough
	bfield[0] = b[0]; 
	bfield[1] = b[1]; 
	bfield[2] = b[2];
	return;
    }

    const GlobalPoint g(xyz[0]/cm,xyz[1]/cm,xyz[2]/cm);
    GlobalVector v = theCMSMagneticField->inTesla(g);
    b[0] = v.x()*tesla;
    b[1] = v.y()*tesla;
    b[2] = v.z()*tesla;

    oldx[0] = xyz[0]; 
    oldx[1] = xyz[1]; 
    oldx[2] = xyz[2];

    bfield[0] = b[0]; 
    bfield[1] = b[1]; 
    bfield[2] = b[2];
}

void Field::fieldEquation(G4Mag_UsualEqRhs * e) { theFieldEquation = e; }

