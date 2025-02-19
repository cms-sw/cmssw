#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Parallelepiped.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Para.hh>
#include <string>

class testParallelepiped : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testParallelepiped );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testParallelepiped::matched_g4_and_dd( void )
{
  double xHalf =  5.*cm;
  double yHalf = 6.*cm;
  double zHalf = 7.*cm;
  double alpha = 15.*deg;
  double theta = 30.*deg;
  double phi = 45.*deg;
  std::string name( "fred1" );

  G4Para g4( name, xHalf, yHalf, zHalf, alpha, theta, phi );
  DDI::Parallelepiped dd(    xHalf, yHalf, zHalf, alpha, theta, phi );
  DDParallelepiped dds = DDSolidFactory::parallelepiped( name, xHalf, yHalf, zHalf, alpha, theta, phi );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;

  double g4_volume = g4.GetCubicVolume()/cm3;
  double dd_volume = dd.volume()/cm3;
  double dds_volume = dds.volume()/cm3;
  
  std::cout << "\tg4 volume = " << g4_volume <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd_volume << " cm3" <<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol=" << dds_volume << " cm3" << std::endl;
  
  CPPUNIT_ASSERT( g4_volume == dd_volume );
  CPPUNIT_ASSERT( g4_volume == dds_volume );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testParallelepiped );
