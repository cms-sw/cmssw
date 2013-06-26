#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Orb.h>
#include <DetectorDescription/Core/src/Sphere.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Orb.hh>
#include <string>

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
  std::string name( "fred1" );

  G4Orb g4( name, radius );
  DDI::Orb dd( radius );
  DDI::Sphere dds( 0.*deg, radius, 0.*deg, 360.*deg, 0., 180.*deg );
  DDOrb ddo = DDSolidFactory::orb( name, radius );
  dd.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << ddo << " vol= " << ddo.volume() << std::endl;
  std::cout << "\tcross check sphere " << std::endl;
  dds.stream(std::cout);
  std::cout << std::endl;
  std::cout << "\tsphere volume = " << dds.volume()/cm3 << " cm3" << std::endl;

  double g4v = g4.GetCubicVolume()/cm3;
  double ddv = dd.volume()/cm3;
  double ddsv = dds.volume()/cm3;
  
  CPPUNIT_ASSERT( g4v == ddv );
  CPPUNIT_ASSERT( g4v == ddsv );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testOrb );
