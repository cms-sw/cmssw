#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Parallelepiped.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Para.hh>
#include <cmath>
#include <string>
#include <limits>

using namespace std;

class testPara : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testPara );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testPara::matched_g4_and_dd( void )
{
  double xHalf =  5.*cm;
  double yHalf = 6.*cm;
  double zHalf = 7.*cm;
  double alpha = 15.*deg;
  double theta = 30.*deg;
  double phi = 45.*deg;
  string name( "fred1" );

  G4Para g4( name, xHalf, yHalf, zHalf, alpha, theta, phi );
  DDI::Parallelepiped dd( xHalf, yHalf, zHalf, alpha, theta, phi );
  DDParallelepiped dds = DDSolidFactory::parallelepiped( name, xHalf, yHalf, zHalf, alpha, theta, phi );
  cout << endl;
  dd.stream( cout );
  cout << endl;

  double g4v = g4.GetCubicVolume()/cm3;
  double ddv = dd.volume()/cm3;
  double ddsv = dds.volume()/cm3;
  
  cout << "\tg4 volume = " << g4v <<" cm3" << endl;
  cout << "\tdd volume = " << ddv << " cm3" <<  endl;
  cout << "\tDD Information: " << dds << " vol=" << ddsv << " cm3" << endl;
  
  CPPUNIT_ASSERT( abs(g4v - ddv) < numeric_limits<float>::epsilon());
  CPPUNIT_ASSERT( abs(g4v - ddsv) < numeric_limits<float>::epsilon());
}

CPPUNIT_TEST_SUITE_REGISTRATION( testPara );
