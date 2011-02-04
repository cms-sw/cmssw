#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Cons.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Cons.hh>
#include <string>

class testCons : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testCons );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testCons::matched_g4_and_dd( void )
{
  double rIn1 = 10.*cm;
  double rOut1 = 15.*cm;
  double rIn2 = 20.*cm;
  double rOut2 = 25.*cm;
  double zhalf = 20.*cm;
  double startPhi = 0.*deg;
  double deltaPhi = 90.*deg;
  std::string name( "fred1" );

  G4Cons g4( name, rIn1, rOut1, rIn2, rOut2, zhalf, startPhi, deltaPhi );
  DDI::Cons dd( zhalf, rIn1, rOut1, rIn2, rOut2, startPhi, deltaPhi );
  DDCons dds = DDSolidFactory::cons( name, zhalf, rIn1, rOut1, rIn2, rOut2, startPhi, deltaPhi );
  std::cout << std::endl;
  dd.stream(std::cout);
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

CPPUNIT_TEST_SUITE_REGISTRATION( testCons );
