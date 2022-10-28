#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/PseudoTrap.h"
#include "SimG4Core/Geometry/interface/DDG4SolidConverter.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4VSolid.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testPseudoTrap : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testPseudoTrap);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testPseudoTrap::matched_g4_and_dd(void) {
  double pDx1 = 0.293734 * m;  /**< Half-length along x at the surface positioned at -dz */
  double pDx2 = 1.86356 * m;   /**<  Half-length along x at the surface positioned at +dz */
  double pDy1 = 0.3000 * m;    /**<  Half-length along y at the surface positioned at -dz */
  double pDy2 = 0.3000 * m;    /**<  Half-length along y at the surface positioned at +dz */
  double pDz = 2.92934 * m;    /**< Half of the height of the pseudo trapezoid along z */
  double radius = -1.1350 * m; /**< radius of the cut-out (negative sign) or rounding (pos. sign) */
  bool atMinusZ = true;        /**< if true, the cut-out or rounding is applied at -dz,
                           else at +dz */
  string name("fred1");

  DDI::PseudoTrap dd(pDx1, pDx2, pDy1, pDy2, pDz, radius, atMinusZ);
  DDPseudoTrap dds = DDSolidFactory::pseudoTrap(name, pDx1, pDx2, pDy1, pDy2, pDz, radius, atMinusZ);
  G4VSolid *g4 = DDG4SolidConverter::pseudotrap(dds);
  cout << endl;
  dd.stream(cout);
  cout << endl;

  double g4v = g4->GetCubicVolume() / cm3;
  double ddv = dd.volume() / cm3;
  double ddsv = dds.volume() / cm3;

  cout << "\tg4 volume = " << g4v << " cm3" << endl;
  cout << "\tdd volume = " << ddv << " cm3" << endl;
  cout << "\tDD Information: " << dds << " vol=" << ddsv << " cm3" << endl;

  if (ddv > 0.0) {
    CPPUNIT_ASSERT(abs(g4v - ddv) < numeric_limits<float>::epsilon());
    CPPUNIT_ASSERT(abs(g4v - ddsv) < numeric_limits<float>::epsilon());
  }
}

CPPUNIT_TEST_SUITE_REGISTRATION(testPseudoTrap);
