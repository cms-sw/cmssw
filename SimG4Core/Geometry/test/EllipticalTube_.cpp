#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/EllipticalTube.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4EllipticalTube.hh>
#include <string>

class testEllipticalTube : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testEllipticalTube );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testEllipticalTube::matched_g4_and_dd( void )
{
  double xSemiAxis = 3.0*cm;
  double ySemiAxis = 2.0*cm;
  double zHeight = 5.0*cm;
  std::string name( "fred1" );

  G4EllipticalTube g4( name, xSemiAxis, ySemiAxis, zHeight );
  DDI::EllipticalTube dd(    xSemiAxis, ySemiAxis, zHeight );
  DDEllipticalTube dds = DDSolidFactory::ellipticalTube( name, xSemiAxis, ySemiAxis, zHeight );
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

CPPUNIT_TEST_SUITE_REGISTRATION( testEllipticalTube );
