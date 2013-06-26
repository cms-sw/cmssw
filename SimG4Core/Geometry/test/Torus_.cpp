#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Torus.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Torus.hh>
#include <string>

class testTorus : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testTorus );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testTorus::matched_g4_and_dd( void )
{
  double rMin = 10.*cm;
  double rMax = 15.*cm;
  double radius = 100.*cm;
  double sPhi = 0.*deg;
  double dPhi = 90.*deg;
  std::string name( "fred1" );

  G4Torus g4( name, rMin, rMax, radius, sPhi, dPhi );
  DDI::Torus dd( rMin, rMax, radius, sPhi, dPhi );
  DDTorus dds = DDSolidFactory::torus( name, rMin, rMax, radius, sPhi, dPhi );
  std::cout << std::endl;
  dd.stream(std::cout);
  std::cout << std::endl;

  double g4v = g4.GetCubicVolume()/cm3;
  double ddv = dd.volume()/cm3;
  double ddsv = dds.volume()/cm3;

  std::cout << "\tg4 volume = " << g4v <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << ddv << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << ddsv << " cm3" << std::endl;
  
  CPPUNIT_ASSERT( g4v == ddv );
  CPPUNIT_ASSERT( g4v == ddsv );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testTorus );
