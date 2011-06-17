#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Tubs.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Tubs.hh>
#include <string>

class testTubs : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testTubs );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testTubs::matched_g4_and_dd( void )
{
  double rIn = 10.*cm;
  double rOut = 15.*cm;
  double zhalf = 20.*cm;
  double startPhi = 0.*deg;
  double deltaPhi = 90.*deg;
  std::string name( "fred1" );

  G4Tubs g4( name, rIn, rOut, zhalf, startPhi, deltaPhi );
  DDI::Tubs dd( zhalf, rIn, rOut, startPhi, deltaPhi );
  DDTubs dds = DDSolidFactory::tubs( name, zhalf, rIn, rOut, startPhi, deltaPhi );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;
  
  double g4v = g4.GetCubicVolume()/cm3;
  double ddv = dd.volume()/cm3;
  double ddsv = dds.volume()/cm3;
  
  std::cout << "\tg4 volume = " << g4v <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << ddv << " cm3" <<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol=" << ddsv << " cm3" << std::endl;

  double tolerance = 1e-7;
  
  CPPUNIT_ASSERT( fabs( g4v - ddv ) < tolerance );
  CPPUNIT_ASSERT( fabs( g4v - ddsv ) < tolerance );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testTubs );
