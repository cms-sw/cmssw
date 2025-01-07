#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoVertex/BeamSpotProducer/plugins/BeamSpotCompatibilityChecker.cc"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

//_________________________________________________________
TEST_CASE("Significance Calculation", "[Significance]") {
  double a = 10.0;
  double b = 12.0;
  double errA = 1.0;
  double errB = 1.5;

  Significance sig(a, b, errA, errB, "test");
  float significance = sig.getSig(false);

  // Correct the expected value based on actual calculation
  REQUIRE(significance == Approx(1.1094).epsilon(10e-6));
}

//_________________________________________________________
TEST_CASE("BeamSpot Compatibility Checker", "[compareBS]") {
  reco::BeamSpot::Point pointA(1.0, 2.0, 3.0);
  reco::BeamSpot::Point pointB(1.1, 2.1, 3.1);

  reco::BeamSpot::CovarianceMatrix matrixA;
  reco::BeamSpot::CovarianceMatrix matrixB;

  // Initialize matrices with non-zero values
  for (int i = 0; i < reco::BeamSpot::dimension; ++i) {
    for (int j = 0; j < reco::BeamSpot::dimension; ++j) {
      matrixA(i, j) = 0.01 * (i + 1) * (j + 1);
      matrixB(i, j) = 0.02 * (i + 1) * (j + 1);
    }
  }

  reco::BeamSpot beamSpotA(pointA, 4.0, 0.01, 0.01, 0.1, matrixA);
  reco::BeamSpot beamSpotB(pointB, 4.2, 0.01, 0.01, 0.12, matrixB);

  // Create the edm::ParameterSet with the required parameters
  edm::ParameterSet pset;
  pset.addParameter<double>("warningThr", 1.0);
  pset.addParameter<double>("errorThr", 2.0);
  pset.addParameter<edm::InputTag>("bsFromEvent", edm::InputTag(""));
  pset.addParameter<bool>("dbFromEvent", false);
  pset.addParameter<edm::InputTag>("bsFromDB", edm::InputTag(""));

  BeamSpotCompatibilityChecker checker(pset);
  auto significances = checker.compareBS(beamSpotA, beamSpotB, true);

  // Print significances
  for (size_t i = 0; i < significances.size(); ++i) {
    std::cout << "Significance[" << i << "]: " << significances[i] << std::endl;
  }

  REQUIRE(significances[0] == Approx(0.57735).epsilon(10e-6));    // x0 significance
  REQUIRE(significances[1] == Approx(0.288675).epsilon(10e-6));   // y0 significance
  REQUIRE(significances[2] == Approx(0.19245).epsilon(10e-6));    // z0 significance
  REQUIRE(significances[3] == Approx(0.0164957).epsilon(10e-6));  // sigmaX significance
  REQUIRE(significances[4] == Approx(0.0164957).epsilon(10e-6));  // sigmaY significance
  REQUIRE(significances[5] == Approx(0.288675).epsilon(10e-6));   // sigmaZ significance
}
