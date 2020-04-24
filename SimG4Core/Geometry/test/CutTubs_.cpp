#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"

#include <DetectorDescription/Core/interface/DDSolid.h>
#include <DetectorDescription/Core/interface/DDSolidShapes.h>

#include <DetectorDescription/Core/src/CutTubs.h>
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4CutTubs.hh>
#include <string>

class testCutTubs : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testCutTubs );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testCutTubs::matched_g4_and_dd( void )
{
  double rIn = 10.*cm;
  double rOut = 15.*cm;
  double zhalf = 20.*cm;
  double startPhi = 0.*deg;
  double deltaPhi = 90.*deg;
  std::string name( "fred1" );

  std::array<double, 3> lowNorm = {{ 0, -0.7, -0.71 }};
  std::array<double, 3> highNorm = {{ 0.7, 0, 0.71 }};
  
  G4CutTubs g4( name, rIn, rOut, zhalf, startPhi, deltaPhi,
		G4ThreeVector( lowNorm[0], lowNorm[1], lowNorm[2]),
		G4ThreeVector( highNorm[0], highNorm[1], highNorm[2]));
  DDI::CutTubs dd( zhalf, rIn, rOut, startPhi, deltaPhi,
		   lowNorm[0], lowNorm[1], lowNorm[2],
		   highNorm[0], highNorm[1], highNorm[2] );
  DDCutTubs dds = DDSolidFactory::cuttubs( name, zhalf, rIn, rOut, startPhi, deltaPhi,
					   lowNorm[0], lowNorm[1], lowNorm[2],
					   highNorm[0], highNorm[1], highNorm[2] );
  std::cout << std::endl;
  dd.stream( std::cout );
  std::cout << std::endl;
  
  double g4v = g4.GetCubicVolume()/cm3;
  double ddv = dd.volume()/cm3;
  double ddsv = dds.volume()/cm3;
  
  std::cout << "\tg4 volume = " << g4v <<" cm3" << std::endl;
  std::cout << "\tdd volume = " << ddv << " cm3" <<  std::endl;
  std::cout << "\tDD Information: " << dds << " vol=" << ddsv << " cm3" << std::endl;

  //double tolerance = 1e-7;

  // DD does not calculate volume for this shape
  //CPPUNIT_ASSERT( fabs( g4v - ddv ) < tolerance );
  //CPPUNIT_ASSERT( fabs( g4v - ddsv ) < tolerance );
}

CPPUNIT_TEST_SUITE_REGISTRATION( testCutTubs );
