#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Box.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Box.hh>
#include <string>

class testBox : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testBox );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testBox::matched_g4_and_dd( void )
{
  double xHalfLength( 2.*cm );
  double yHalfLength( 2.*cm );
  double zHalfLength( 2.*cm );
  std::string name( "fred1" );

  G4Box g4( name, xHalfLength, yHalfLength, zHalfLength );
  DDI::Box dd( xHalfLength, yHalfLength, zHalfLength );
  DDBox dds = DDSolidFactory::box( name, xHalfLength, yHalfLength, zHalfLength );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;
  std::cout << "\tg4 volume = " << g4.GetCubicVolume()/cm3 <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << dd.volume()/cm3 << " cm3"<<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol= " << dds.volume() << std::endl;
  
  CPPUNIT_ASSERT( g4.GetCubicVolume()/cm3 == dd.volume()/cm3 );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testBox );
