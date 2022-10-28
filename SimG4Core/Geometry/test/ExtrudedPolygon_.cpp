#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/ExtrudedPolygon.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4ExtrudedSolid.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testExtrudedPgon : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testExtrudedPgon);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testExtrudedPgon::matched_g4_and_dd(void) {
  vector<double> x = {-300, -300, 300, 300, 150, 150, -150, -150};
  vector<double> y = {-300, 300, 300, -300, -300, 150, 150, -300};
  vector<double> z = {-600, -150, 100, 600};
  vector<double> zx = {0, 0, 0, 0};
  vector<double> zy = {300, -300, 0, 300};
  vector<double> zscale = {8, 10, 6, 12};
  string name("fred1");

  vector<G4TwoVector> polygon;
  vector<G4ExtrudedSolid::ZSection> zsections;
  for (unsigned int it = 0; it < x.size(); ++it)
    polygon.emplace_back(x[it], y[it]);
  for (unsigned int it = 0; it < z.size(); ++it)
    zsections.emplace_back(z[it], G4TwoVector(zx[it], zy[it]), zscale[it]);
  G4ExtrudedSolid g4(name, polygon, zsections);
  DDI::ExtrudedPolygon dd(x, y, z, zx, zy, zscale);
  DDExtrudedPolygon dds = DDSolidFactory::extrudedpolygon(name, x, y, z, zx, zy, zscale);

  dd.stream(cout);
  cout << endl;
  cout << "\tg4 volume = " << g4.GetCubicVolume() / cm3 << " cm3" << endl;
  cout << "\tdd volume = " << dd.volume() / cm3 << " cm3" << endl;
  cout << "\tDD Information: " << dds << " vol= " << dds.volume() << endl;

  // FIXME: dd voulme is not implemented yet!
}

CPPUNIT_TEST_SUITE_REGISTRATION(testExtrudedPgon);
