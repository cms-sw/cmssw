#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Parallelepiped.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Para.hh>
#include <string>

class testPara : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testPara );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( );
};

void
testPara::matched_g4_and_dd( )
{
  double xHalf =  5.*cm;
  double yHalf = 6.*cm;
  double zHalf = 7.*cm;
  double alpha = 15.*deg;
  double theta = 30.*deg;
  double phi = 45.*deg;
  std::string name( "fred1" );

  G4Para g4( name, xHalf, yHalf, zHalf, alpha, theta, phi );
  DDI::Parallelepiped dd( xHalf, yHalf, zHalf, alpha, theta, phi );
  DDParallelepiped dds = DDSolidFactory::parallelepiped( name, xHalf, yHalf, zHalf, alpha, theta, phi );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;

  double g4v = g4.GetCubicVolume()/cm3;
  double ddv = dd.volume()/cm3;
  double ddsv = dds.volume()/cm3;
  
  std::cout << "\tg4 volume = " << g4v <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << ddv << " cm3" <<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol=" << ddsv << " cm3" << std::endl;
  
  CPPUNIT_ASSERT( g4v == ddv );
  CPPUNIT_ASSERT( g4v == ddsv );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testPara );
