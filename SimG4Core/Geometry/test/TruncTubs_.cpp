#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/TruncTubs.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4VSolid.hh>

#include <string>
#include <SimG4Core/Geometry/interface/DDG4SolidConverter.h>

class testTruncTubs : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testTruncTubs );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testTruncTubs::matched_g4_and_dd( void )
{
  double zHalf = 50.0*cm;
  double rIn = 20.0*cm;
  double rOut = 40.0*cm;
  double startPhi = 0.0*deg;
  double deltaPhi = 90.0*deg;
  double cutAtStart = 25.0*cm;
  double cutAtDelta = 35.0*cm;
  bool cutInside = true;
  std::string name( "fred1" );

  DDI::TruncTubs dd( zHalf, rIn, rOut, startPhi, deltaPhi, cutAtStart, cutAtDelta, cutInside );
  DDTruncTubs dds = DDSolidFactory::truncTubs( name, zHalf, rIn, rOut, startPhi, deltaPhi, cutAtStart, cutAtDelta, cutInside );
  G4VSolid *g4 = DDG4SolidConverter::trunctubs( dds );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;

  double g4_volume = g4->GetCubicVolume()/cm3;
  double dd_volume = dd.volume()/cm3;
  double dds_volume = dds.volume()/cm3;
  
  std::cout << "\tg4 volume = " << g4_volume <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd_volume << " cm3" <<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol=" << dds_volume << " cm3" << std::endl;

  if( dd_volume > 0 )
  {
    CPPUNIT_ASSERT( g4_volume == dd_volume );
    CPPUNIT_ASSERT( g4_volume == dds_volume );
  }
}

CPPUNIT_TEST_SUITE_REGISTRATION( testTruncTubs );
