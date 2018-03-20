#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Orb.h"
#include "DetectorDescription/Core/src/Sphere.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Orb.hh>
#include <cmath>
#include <string>
#include <limits>

using namespace std;

class testOrb : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testOrb );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testOrb::matched_g4_and_dd( void )
{
  double radius = 10.*cm;
  string name( "fred1" );

  G4Orb g4( name, radius );
  DDI::Orb dd( radius );
  DDI::Sphere dds( 0.*deg, radius, 0.*deg, 360.*deg, 0., 180.*deg );
  DDOrb ddo = DDSolidFactory::orb( name, radius );
  dd.stream(cout);
  cout << endl;
  cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << endl;
  cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  endl;
  cout << "\tDD Information: " << ddo << " vol= " << ddo.volume() << endl;
  cout << "\tcross check sphere " << endl;
  dds.stream(cout);
  cout << endl;
  cout << "\tsphere volume = " << dds.volume()/cm3 << " cm3" << endl;

  double g4v = g4.GetCubicVolume()/cm3;
  double ddv = dd.volume()/cm3;
  double ddsv = dds.volume()/cm3;
  
  CPPUNIT_ASSERT( abs(g4v - ddv) < numeric_limits<float>::epsilon());
  CPPUNIT_ASSERT( abs(g4v - ddsv) < numeric_limits<float>::epsilon());
}

CPPUNIT_TEST_SUITE_REGISTRATION( testOrb );
