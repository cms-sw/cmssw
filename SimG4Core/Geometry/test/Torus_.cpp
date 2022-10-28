#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Torus.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4Torus.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testTorus : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTorus);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testTorus::matched_g4_and_dd(void) {
  double rMin = 10. * cm;
  double rMax = 15. * cm;
  double radius = 100. * cm;
  double sPhi = 0. * deg;
  double dPhi = 90. * deg;
  string name("fred1");

  G4Torus g4(name, rMin, rMax, radius, sPhi, dPhi);
  DDI::Torus dd(rMin, rMax, radius, sPhi, dPhi);
  DDTorus dds = DDSolidFactory::torus(name, rMin, rMax, radius, sPhi, dPhi);
  cout << endl;
  dd.stream(cout);
  cout << endl;

  double g4v = g4.GetCubicVolume() / cm3;
  double ddv = dd.volume() / cm3;
  double ddsv = dds.volume() / cm3;

  cout << "\tg4 volume = " << g4v << " cm3" << endl;
  cout << "\tdd volume = " << ddv << " cm3" << endl;
  cout << "\tDD Information: " << dds << " vol= " << ddsv << " cm3" << endl;

  CPPUNIT_ASSERT(abs(g4v - ddv) < numeric_limits<float>::epsilon());
  CPPUNIT_ASSERT(abs(g4v - ddsv) < numeric_limits<float>::epsilon());
}

CPPUNIT_TEST_SUITE_REGISTRATION(testTorus);
