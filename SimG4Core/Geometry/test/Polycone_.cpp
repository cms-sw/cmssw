#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Polycone.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Polycone.hh>
#include <string>

class testPolycone : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testPolycone );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testPolycone::matched_g4_and_dd( void )
{
  double phiStart = 45.*deg;
  double phiTotal = 325.*deg;
  double inner[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  std::vector<double> rInner( inner, inner + sizeof( inner ) / sizeof( double ));
  double outer[] = { 0, 10, 10, 5, 5, 10, 10, 2, 2 };
  std::vector<double> rOuter( outer, outer + sizeof( outer ) / sizeof( double ));
  double pl[] = { 5, 7, 9, 11, 25, 27, 29, 31, 35 };
  std::vector<double> z( pl, pl + sizeof( pl ) / sizeof( double ));
  std::string name( "fred1" );

  G4Polycone g4( name, phiStart, phiTotal, z.size(), &z[0], &rInner[0], &rOuter[0] );
  DDI::Polycone dd( phiStart, phiTotal, z, rInner, rOuter );
  DDPolycone dds = DDSolidFactory::polycone( name, phiStart, phiTotal, z, rInner, rOuter );
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

CPPUNIT_TEST_SUITE_REGISTRATION( testPolycone );
