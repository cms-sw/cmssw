#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/EllipticalTube.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4EllipticalTube.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testEllipticalTube : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEllipticalTube);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testEllipticalTube::matched_g4_and_dd(void) {
  double xSemiAxis = 3.0 * cm;
  double ySemiAxis = 2.0 * cm;
  double zHeight = 5.0 * cm;
  string name("fred1");

  G4EllipticalTube g4(name, xSemiAxis, ySemiAxis, zHeight);
  DDI::EllipticalTube dd(xSemiAxis, ySemiAxis, zHeight);
  DDEllipticalTube dds = DDSolidFactory::ellipticalTube(name, xSemiAxis, ySemiAxis, zHeight);
  cout << endl;
  dd.stream(cout);
  cout << endl;

  double g4v = g4.GetCubicVolume() / cm3;
  double ddv = dd.volume() / cm3;
  double ddsv = dds.volume() / cm3;

  cout << "\tg4 volume = " << g4v << " cm3" << endl;
  cout << "\tdd volume = " << ddv << " cm3" << endl;
  cout << "\tDD Information: " << dds << " vol=" << ddsv << " cm3" << endl;

  CPPUNIT_ASSERT(abs(g4v - ddv) < numeric_limits<float>::epsilon());
  CPPUNIT_ASSERT(abs(g4v - ddsv) < numeric_limits<float>::epsilon());
}

CPPUNIT_TEST_SUITE_REGISTRATION(testEllipticalTube);
