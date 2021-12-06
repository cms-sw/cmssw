#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DetectorDescription/Core/interface/CutTubs.h"
#include "Utilities/Testing/interface/CppUnit_testdriver.icpp"
#include "cppunit/extensions/HelperMacros.h"
#include <G4CutTubs.hh>
#include <cmath>
#include <limits>
#include <string>

using namespace std;

class testCutTubs : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testCutTubs);
  CPPUNIT_TEST(matched_g4_and_dd);

  CPPUNIT_TEST_SUITE_END();

public:
  void matched_g4_and_dd(void);
};

void testCutTubs::matched_g4_and_dd(void) {
  double rIn = 10. * cm;
  double rOut = 15. * cm;
  double zhalf = 20. * cm;
  double startPhi = 0. * deg;
  double deltaPhi = 90. * deg;
  string name("fred1");

  array<double, 3> lowNorm = {{0, -0.7, -0.71}};
  array<double, 3> highNorm = {{0.7, 0, 0.71}};

  G4CutTubs g4(name,
               rIn,
               rOut,
               zhalf,
               startPhi,
               deltaPhi,
               G4ThreeVector(lowNorm[0], lowNorm[1], lowNorm[2]),
               G4ThreeVector(highNorm[0], highNorm[1], highNorm[2]));
  DDI::CutTubs dd(
      zhalf, rIn, rOut, startPhi, deltaPhi, lowNorm[0], lowNorm[1], lowNorm[2], highNorm[0], highNorm[1], highNorm[2]);
  DDCutTubs dds = DDSolidFactory::cuttubs(name,
                                          zhalf,
                                          rIn,
                                          rOut,
                                          startPhi,
                                          deltaPhi,
                                          lowNorm[0],
                                          lowNorm[1],
                                          lowNorm[2],
                                          highNorm[0],
                                          highNorm[1],
                                          highNorm[2]);
  cout << endl;
  dd.stream(cout);
  cout << endl;

  double g4v = g4.GetCubicVolume() / cm3;
  double ddv = dd.volume() / cm3;
  double ddsv = dds.volume() / cm3;

  cout << "\tg4 volume = " << g4v << " cm3" << endl;
  cout << "\tdd volume = " << ddv << " cm3" << endl;
  cout << "\tDD Information: " << dds << " vol=" << ddsv << " cm3" << endl;
}

CPPUNIT_TEST_SUITE_REGISTRATION(testCutTubs);
