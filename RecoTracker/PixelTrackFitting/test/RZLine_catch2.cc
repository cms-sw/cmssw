#define CATCH_CONFIG_MAIN
#include <catch.hpp>
#include <algorithm>
#include "RecoTracker/PixelTrackFitting/interface/RZLine.h"

TEST_CASE("test RZLine", "[RZLine]") {
  SECTION("Constructors") {
    constexpr std::array<float, 2> r{{0.001, 0.003}};
    constexpr std::array<float, 2> z{{0.1, 0.2}};
    constexpr std::array<float, 2> errz{{0.01, 0.02}};
    constexpr float cotTheta = 50.0f;
    constexpr float intercept = 0.05f;
    constexpr float covss = 124.99996f;
    constexpr float covii = 0.0003249999136f;
    constexpr float covsi = -0.175f;
    constexpr float chi2 = 0;
    constexpr float chi2_diff = 1e-10;

    SECTION("array") {
      RZLine l(r, z, errz);

      REQUIRE(l.cotTheta() == Approx(cotTheta));
      REQUIRE(l.intercept() == Approx(intercept));
      REQUIRE(l.covss() == Approx(covss));
      REQUIRE(l.covii() == Approx(covii));
      REQUIRE(l.covsi() == Approx(covsi));
      REQUIRE_THAT(l.chi2(), Catch::Matchers::WithinAbs(chi2, chi2_diff));
    }

    SECTION("vector") {
      const std::vector<float> rv{r.begin(), r.end()};
      const std::vector<float> zv{z.begin(), z.end()};
      const std::vector<float> errzv{errz.begin(), errz.end()};
      RZLine l(rv, zv, errzv);

      REQUIRE(l.cotTheta() == Approx(cotTheta));
      REQUIRE(l.intercept() == Approx(intercept));
      REQUIRE(l.covss() == Approx(covss));
      REQUIRE(l.covii() == Approx(covii));
      REQUIRE(l.covsi() == Approx(covsi));
      REQUIRE_THAT(l.chi2(), Catch::Matchers::WithinAbs(chi2, chi2_diff));
    }

    SECTION("vector ErrZ2_tag") {
      const std::vector<float> rv{r.begin(), r.end()};
      const std::vector<float> zv{z.begin(), z.end()};
      std::vector<float> errzv;
      std::transform(errz.begin(), errz.end(), std::back_inserter(errzv), [](float v) { return v * v; });
      RZLine l(rv, zv, errzv, RZLine::ErrZ2_tag());

      REQUIRE(l.cotTheta() == Approx(cotTheta));
      REQUIRE(l.intercept() == Approx(intercept));
      REQUIRE(l.covss() == Approx(covss));
      REQUIRE(l.covii() == Approx(covii));
      REQUIRE(l.covsi() == Approx(covsi));
      REQUIRE_THAT(l.chi2(), Catch::Matchers::WithinAbs(chi2, chi2_diff));
    }

    SECTION("points&errors") {
      struct Error {
        float err_;
        float czz() const { return err_ * err_; }
        float rerr(const GlobalPoint&) const { return 0.f; }
      };
      const std::vector<GlobalPoint> p{{{std::abs(r[0]), 0., z[0]}, {std::abs(r[1]), 0., z[1]}}};
      REQUIRE(p[0].perp() == std::abs(r[0]));
      REQUIRE(p[0].z() == z[0]);
      REQUIRE(p[1].perp() == std::abs(r[1]));
      REQUIRE(p[1].z() == z[1]);
      const std::vector<Error> err{{{errz[0]}, {errz[1]}}};
      REQUIRE(err[0].czz() == errz[0] * errz[0]);
      REQUIRE(err[1].czz() == errz[1] * errz[1]);
      const std::vector<bool> barrel{{true, true}};
      RZLine l(p, err, barrel);

      REQUIRE(l.cotTheta() == Approx(cotTheta));
      REQUIRE(l.intercept() == Approx(intercept));
      REQUIRE(l.covss() == Approx(covss));
      REQUIRE(l.covii() == Approx(covii));
      REQUIRE(l.covsi() == Approx(covsi));
      REQUIRE_THAT(l.chi2(), Catch::Matchers::WithinAbs(chi2, chi2_diff));
    }
  }
  SECTION("no points") {
    const std::vector<float> rv;
    const std::vector<float> zv;
    std::vector<float> errzv;
    RZLine l(rv, zv, errzv);

    //NOTE a NaN is a number that is not equal to itself
    REQUIRE(l.cotTheta() != l.cotTheta());
    REQUIRE(l.intercept() != l.intercept());
    REQUIRE(l.covss() != l.covss());
    REQUIRE(l.covii() != l.covii());
    REQUIRE(l.covsi() != l.covsi());
    REQUIRE(l.chi2() == 0.0f);
  }
  SECTION("one point") {
    const std::vector<float> rv = {0.001};
    const std::vector<float> zv = {0.1};
    const std::vector<float> errzv = {0.01};
    RZLine l(rv, zv, errzv);

    //NOTE a NaN is a number that is not equal to itself
    REQUIRE(l.cotTheta() != l.cotTheta());
    REQUIRE(l.intercept() != l.intercept());
    REQUIRE(l.covss() != l.covss());
    REQUIRE(l.covii() != l.covii());
    REQUIRE(l.covsi() != l.covsi());
    REQUIRE(l.chi2() == 0.0f);
  }
}
