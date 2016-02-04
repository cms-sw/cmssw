#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/Sphere.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Sphere.hh>
#include <string>

class testSphere : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testSphere );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testSphere::matched_g4_and_dd( void )
{
  double innerRadius = 10.*cm;
  double outerRadius = 15.*cm;
  double startPhi = 0.*deg;
  double deltaPhi = 90.*deg;
  double startTheta = 0.*deg;
  double deltaTheta = 180.*deg;
  std::string name( "fred1" );

  G4Sphere g4( name, innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta );
  DDI::Sphere dd( innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta );
  DDSphere dds = DDSolidFactory::sphere( name, innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta );
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

CPPUNIT_TEST_SUITE_REGISTRATION( testSphere );
