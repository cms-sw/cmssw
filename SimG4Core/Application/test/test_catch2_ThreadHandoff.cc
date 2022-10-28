#include "SimG4Core/Application/interface/ThreadHandoff.h"
#include "FWCore/Utilities/interface/Exception.h"
#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace omt;
TEST_CASE("Test omt::ThreadHandoff", "[ThreadHandoff]") {
  SECTION("Do nothing") { ThreadHandoff th; }
  SECTION("Simple") {
    ThreadHandoff th;
    bool value = false;
    th.runAndWait([&value]() { value = true; });
    REQUIRE(value == true);
  }

  SECTION("Exception") {
    ThreadHandoff th;
    REQUIRE_THROWS_AS(th.runAndWait([]() { throw cms::Exception("Test"); }), cms::Exception);
  }
}
