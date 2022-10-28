#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Sphere.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4Sphere.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testSphere : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testSphere);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testSphere::matched_g4_and_dd(void) {
  double innerRadius = 10. * cm;
  double outerRadius = 15. * cm;
  double startPhi = 0. * deg;
  double deltaPhi = 90. * deg;
  double startTheta = 0. * deg;
  double deltaTheta = 180. * deg;
  string name("fred1");

  G4Sphere g4(name, innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
  DDI::Sphere dd(innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
  DDSphere dds = DDSolidFactory::sphere(name, innerRadius, outerRadius, startPhi, deltaPhi, startTheta, deltaTheta);
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

CPPUNIT_TEST_SUITE_REGISTRATION(testSphere);
