#include "SimG4Core/MagneticField/interface/Field.h"
#include "MagneticField/Engine/interface/MagneticField.h"

//#include "Geometry/Vector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "G4Mag_UsualEqRhs.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include "SimG4Core/Notification/interface/SimG4Exception.h"
#include "FWCore/Utilities/interface/isFinite.h"

using namespace sim;

G4Mag_UsualEqRhs * Field::fieldEquation() { return theFieldEquation; }

Field::Field(const MagneticField * f, double d) 
    : G4MagneticField(), theCMSMagneticField(f),theDelta(d)
{
}

Field::~Field() {}

void Field::GetFieldValue(const double xyz[3],double bfield[3]) const 
{ 

    //
    // this is another trick to check on a NaN, maybe it's even CPU-faster...
    // but ler's stick to system function edm::isNotFinite(...) for now
    //
    // if ( !(xyz[0]==xyz[0]) || !(xyz[1]==xyz[1]) || !(xyz[2]==xyz[2]) )
    if ( edm::isNotFinite(xyz[0]+xyz[1]+xyz[2]) != 0 )
    {
       throw SimG4Exception( "SimG4CoreMagneticField: Corrupted Event - NaN detected (position)" ) ;
    }
    
    // Which is worse, thread_local static here, or mutable members?
    // Mutable members + synchronization would be the most correct. In
    // practice I know there is (by construction via
    // RunManagerMT/RunManagerMTWorker) one Field object per thread,
    // managed via pointers in TLS, so thread_local here is also
    // "correct", and fastest to write.
    static thread_local float oldx[3] = {1.0e12,1.0e12,1.0e12};
    static thread_local double b[3];

    if (theDelta>0. &&
	fabs(oldx[0]-xyz[0])<theDelta &&
	fabs(oldx[1]-xyz[1])<theDelta &&
	fabs(oldx[2]-xyz[2])<theDelta) 
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

void Field::fieldEquation(G4Mag_UsualEqRhs* e) { theFieldEquation = e; }

