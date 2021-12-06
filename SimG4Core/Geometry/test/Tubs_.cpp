#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Tubs.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4Tubs.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testTubs : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTubs);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testTubs::matched_g4_and_dd(void) {
  double rIn = 10. * cm;
  double rOut = 15. * cm;
  double zhalf = 20. * cm;
  double startPhi = 0. * deg;
  double deltaPhi = 90. * deg;
  string name("fred1");

  G4Tubs g4(name, rIn, rOut, zhalf, startPhi, deltaPhi);
  DDI::Tubs dd(zhalf, rIn, rOut, startPhi, deltaPhi);
  DDTubs dds = DDSolidFactory::tubs(name, zhalf, rIn, rOut, startPhi, deltaPhi);
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

CPPUNIT_TEST_SUITE_REGISTRATION(testTubs);
