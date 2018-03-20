#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/src/Ellipsoid.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <G4Ellipsoid.hh>
#include <cmath>
#include <string>
#include <limits>

using namespace std;

class testEllipsoid : public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE( testEllipsoid );
  CPPUNIT_TEST( matched_g4_and_dd );
  
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void matched_g4_and_dd( void );
};

void
testEllipsoid::matched_g4_and_dd( void )
{
  double xSemiAxis = 3.0*cm;
  double ySemiAxis = 2.0*cm;
  double zSemiAxis = 5.0*cm;
  double zBottomCut = 0.0*cm;
  double zTopCut = 0.0*cm;
  string name( "fred1" );

  G4Ellipsoid g4( name, xSemiAxis, ySemiAxis, zSemiAxis, zBottomCut, zTopCut );
  DDI::Ellipsoid dd(    xSemiAxis, ySemiAxis, zSemiAxis, zBottomCut, zTopCut );
  DDEllipsoid dds = DDSolidFactory::ellipsoid( name, xSemiAxis, ySemiAxis, zSemiAxis, zBottomCut, zTopCut );
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

CPPUNIT_TEST_SUITE_REGISTRATION( testEllipsoid );
