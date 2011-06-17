#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Trap.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Trap.hh>
#include <string>

class testTrap : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testTrap );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testTrap::matched_g4_and_dd( void )
{
  double dz = 60.*cm;
  double pTheta = 0.*deg;
  double pPhi = 0.*deg;
  double pDy1 = 30.*cm;
  double pDx1 = 30.*cm;
  double pDx2 = 30.*cm;
  double pAlp1 = 0.*deg;
  double pDy2 = 15.*cm;
  double pDx3 = 10.*cm;
  double pDx4 = 10.*cm;
  double pAlp2 = 0.*deg;
  std::string name( "fred1" );
  // <Trapezoid name="E8CD" dz="17.35*cm" alp1="0*deg" bl1="10.5446*cm" tl1="10.5446*cm" h1="500*mum" alp2="0*deg" bl2="0.1*mum" tl2="0.1*mum" h2="500*mum" phi="180*deg" theta="16.90296*deg" />
  G4Trap g4( name, dz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2 );
  DDI::Trap dd(    dz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2 );
  DDTrap dds = DDSolidFactory::trap( name, dz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2 );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;

  double g4_volume = g4.GetCubicVolume()/cm3;
  double dd_volume = dd.volume()/cm3;
  double dds_volume = dds.volume()/cm3;
  
  std::cout << "\tg4 volume = " << g4_volume <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd_volume << " cm3" <<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol=" << dds_volume << " cm3" << std::endl;
  
  double tolerance = g4_volume;
  
  CPPUNIT_ASSERT( fabs( g4_volume - dd_volume ) < tolerance );
  CPPUNIT_ASSERT( fabs( g4_volume - dds_volume ) < tolerance );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testTrap );
