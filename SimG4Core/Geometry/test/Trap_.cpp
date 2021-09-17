#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Trap.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4Trap.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testTrap : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTrap);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testTrap::matched_g4_and_dd(void) {
  double dz = 60. * cm;
  double pTheta = 0. * deg;
  double pPhi = 0. * deg;
  double pDy1 = 30. * cm;
  double pDx1 = 30. * cm;
  double pDx2 = 30. * cm;
  double pAlp1 = 0. * deg;
  double pDy2 = 15. * cm;
  double pDx3 = 10. * cm;
  double pDx4 = 10. * cm;
  double pAlp2 = 0. * deg;
  string name("fred1");
  // <Trapezoid name="E8CD" dz="17.35*cm" alp1="0*deg" bl1="10.5446*cm"
  // tl1="10.5446*cm" h1="500*mum" alp2="0*deg" bl2="0.1*mum" tl2="0.1*mum"
  // h2="500*mum" phi="180*deg" theta="16.90296*deg" />
  G4Trap g4(name, dz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2);
  DDI::Trap dd(dz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2);
  DDTrap dds = DDSolidFactory::trap(name, dz, pTheta, pPhi, pDy1, pDx1, pDx2, pAlp1, pDy2, pDx3, pDx4, pAlp2);
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

CPPUNIT_TEST_SUITE_REGISTRATION(testTrap);
