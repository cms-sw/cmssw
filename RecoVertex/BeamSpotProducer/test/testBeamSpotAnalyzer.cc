#include "FWCore/TestProcessor/interface/TestProcessor.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

TEST_CASE("BeamSpotAnalyzer tests", "[BeamSpotAnalyzer]") {
  //The python configuration
  edm::test::TestProcessor::Config config{
      R"_(from FWCore.TestProcessor.TestProcess import *
from RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff import d0_phi_analyzer 
process = TestProcess()
process.beamAnalyzer = d0_phi_analyzer 
process.moduleToTest(process.beamAnalyzer)
)_"};

  SECTION("Run with no Lumis") {
    edm::test::TestProcessor tester{config};
    tester.testRunWithNoLuminosityBlocks();
    //get here without an exception or crashing
    REQUIRE(true);
  };
}
