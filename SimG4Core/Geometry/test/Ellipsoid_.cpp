#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Ellipsoid.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Ellipsoid.hh>
#include <string>

class testEllipsoid : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testEllipsoid );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testEllipsoid::matched_g4_and_dd( void )
{
  double xSemiAxis = 3.0*cm;
  double ySemiAxis = 2.0*cm;
  double zSemiAxis = 5.0*cm;
  double zBottomCut = 0.0*cm;
  double zTopCut = 0.0*cm;
  std::string name( "fred1" );

  G4Ellipsoid g4( name, xSemiAxis, ySemiAxis, zSemiAxis, zBottomCut, zTopCut );
  DDI::Ellipsoid dd(    xSemiAxis, ySemiAxis, zSemiAxis, zBottomCut, zTopCut );
  DDEllipsoid dds = DDSolidFactory::ellipsoid( name, xSemiAxis, ySemiAxis, zSemiAxis, zBottomCut, zTopCut );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;

  double g4_volume = g4.GetCubicVolume()/cm3;
  double dd_volume = dd.volume()/cm3;
  double dds_volume = dds.volume()/cm3;
  
  std::cout << "\tg4 volume = " << g4_volume <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd_volume << " cm3" <<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol=" << dds_volume << " cm3" << std::endl;

  double tolerance = 1e-7;
  
  CPPUNIT_ASSERT( fabs( g4_volume - dd_volume ) < tolerance );
  CPPUNIT_ASSERT( fabs( g4_volume - dds_volume ) < tolerance );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testEllipsoid );
