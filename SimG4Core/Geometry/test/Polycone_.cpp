#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Polycone.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4Polycone.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testPolycone : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testPolycone);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testPolycone::matched_g4_and_dd(void) {
  double phiStart = 45. * deg;
  double phiTotal = 325. * deg;
  double inner[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  vector<double> rInner(inner, inner + sizeof(inner) / sizeof(double));
  double outer[] = {0, 10, 10, 5, 5, 10, 10, 2, 2};
  vector<double> rOuter(outer, outer + sizeof(outer) / sizeof(double));
  double pl[] = {5, 7, 9, 11, 25, 27, 29, 31, 35};
  vector<double> z(pl, pl + sizeof(pl) / sizeof(double));
  string name("fred1");

  G4Polycone g4(name, phiStart, phiTotal, z.size(), &z[0], &rInner[0], &rOuter[0]);
  DDI::Polycone dd(phiStart, phiTotal, z, rInner, rOuter);
  DDPolycone dds = DDSolidFactory::polycone(name, phiStart, phiTotal, z, rInner, rOuter);
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

CPPUNIT_TEST_SUITE_REGISTRATION(testPolycone);
