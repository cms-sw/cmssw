#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/Box.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4Box.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testBox : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testBox);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testBox::matched_g4_and_dd(void) {
  double xHalfLength(2. * cm);
  double yHalfLength(2. * cm);
  double zHalfLength(2. * cm);
  string name("fred1");

  G4Box g4(name, xHalfLength, yHalfLength, zHalfLength);
  DDI::Box dd(xHalfLength, yHalfLength, zHalfLength);
  DDBox dds = DDSolidFactory::box(name, xHalfLength, yHalfLength, zHalfLength);
  cout << endl;
  dd.stream(cout);
  cout << endl;

  double g4v = g4.GetCubicVolume() / cm3;
  double ddv = dd.volume() / cm3;
  double ddsv = dds.volume() / cm3;

  cout << "\tg4 volume = " << g4.GetCubicVolume() / cm3 << " cm3" << endl;
  cout << "\tdd volume = " << dd.volume() / cm3 << " cm3" << endl;
  cout << "\tDD Information: " << dds << " vol= " << dds.volume() << endl;

  CPPUNIT_ASSERT(abs(g4v - ddv) < numeric_limits<float>::epsilon());
  CPPUNIT_ASSERT(abs(g4v - ddsv) < numeric_limits<float>::epsilon());
}

CPPUNIT_TEST_SUITE_REGISTRATION(testBox);
