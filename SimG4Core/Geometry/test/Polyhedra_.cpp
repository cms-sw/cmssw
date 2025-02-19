#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Polyhedra.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Polyhedra.hh>
#include <string>

class testPolyhedra : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testPolyhedra );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testPolyhedra::matched_g4_and_dd( void )
{
  int sides = 3;
  double phiStart = 45.*deg;
  double phiTotal = 325.*deg;
  double inner[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  std::vector<double> rInner( inner, inner + sizeof( inner ) / sizeof( double ));
  double outer[] = { 0, 10, 10, 5, 5, 10, 10, 2, 2 };
  std::vector<double> rOuter( outer, outer + sizeof( outer ) / sizeof( double ));
  double pl[] = { 5, 7, 9, 11, 25, 27, 29, 31, 35 };
  std::vector<double> z( pl, pl + sizeof( pl ) / sizeof( double ));
  std::string name( "fred1" );

  G4Polyhedra g4( name, phiStart, phiTotal, sides, z.size(), &z[0], &rInner[0], &rOuter[0] );
  DDI::Polyhedra dd( sides, phiStart, phiTotal, z, rInner, rOuter );
  DDPolyhedra dds = DDSolidFactory::polyhedra( name, sides, phiStart, phiTotal, z, rInner, rOuter );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;

  double g4_volume = g4.GetCubicVolume()/cm3;
  double dd_volume = dd.volume()/cm3;
  double dds_volume = dds.volume()/cm3;
  
  std::cout << "\tg4 volume = " << g4_volume <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd_volume << " cm3" <<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol=" << dds_volume << " cm3" << std::endl;

  double tolerance = 3.5;
  
  CPPUNIT_ASSERT( fabs( g4_volume - dd_volume ) < tolerance );
  CPPUNIT_ASSERT( fabs( g4_volume - dds_volume ) < tolerance );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testPolyhedra );
