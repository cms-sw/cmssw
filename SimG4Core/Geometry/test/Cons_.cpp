#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Cons.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4Cons.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testCons : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCons);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testCons::matched_g4_and_dd(void) {
  double rIn1 = 10. * cm;
  double rOut1 = 15. * cm;
  double rIn2 = 20. * cm;
  double rOut2 = 25. * cm;
  double zhalf = 20. * cm;
  double startPhi = 0. * deg;
  double deltaPhi = 90. * deg;
  string name("fred1");

  G4Cons g4(name, rIn1, rOut1, rIn2, rOut2, zhalf, startPhi, deltaPhi);
  DDI::Cons dd(zhalf, rIn1, rOut1, rIn2, rOut2, startPhi, deltaPhi);
  DDCons dds = DDSolidFactory::cons(name, zhalf, rIn1, rOut1, rIn2, rOut2, startPhi, deltaPhi);
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

CPPUNIT_TEST_SUITE_REGISTRATION(testCons);
