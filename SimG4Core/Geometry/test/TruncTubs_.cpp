#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/TruncTubs.h"
#include "SimG4Core/Geometry/interface/DDG4SolidConverter.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4VSolid.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testTruncTubs : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testTruncTubs);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testTruncTubs::matched_g4_and_dd(void) {
  double zHalf = 50.0 * cm;
  double rIn = 20.0 * cm;
  double rOut = 40.0 * cm;
  double startPhi = 0.0 * deg;
  double deltaPhi = 90.0 * deg;
  double cutAtStart = 25.0 * cm;
  double cutAtDelta = 35.0 * cm;
  bool cutInside = true;
  string name("fred1");

  DDI::TruncTubs dd(zHalf, rIn, rOut, startPhi, deltaPhi, cutAtStart, cutAtDelta, cutInside);
  DDTruncTubs dds =
      DDSolidFactory::truncTubs(name, zHalf, rIn, rOut, startPhi, deltaPhi, cutAtStart, cutAtDelta, cutInside);
  G4VSolid *g4 = DDG4SolidConverter::trunctubs(dds);
  cout << endl;
  dd.stream(cout);
  cout << endl;

  double g4v = g4->GetCubicVolume() / cm3;
  double ddv = dd.volume() / cm3;
  double ddsv = dds.volume() / cm3;

  cout << "\tg4 volume = " << g4v << " cm3" << endl;
  cout << "\tdd volume = " << ddv << " cm3" << endl;
  cout << "\tDD Information: " << dds << " vol=" << ddsv << " cm3" << endl;

  if (ddv > 0) {
    CPPUNIT_ASSERT(abs(g4v - ddv) < numeric_limits<float>::epsilon());
    CPPUNIT_ASSERT(abs(g4v - ddsv) < numeric_limits<float>::epsilon());
  }
}

CPPUNIT_TEST_SUITE_REGISTRATION(testTruncTubs);
