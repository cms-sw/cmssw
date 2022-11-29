#include "catch.hpp"
#include "SimG4CMS/Calo/interface/HFFibre.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

TEST_CASE("test HFFibre", "[HFFibre]") {
  HFFibre::Params params;
  params.fractionOfSpeedOfLightInFibre_ = 0.5;
  params.gParHF_ = {{220, 1650, 300, 0, 11150, 3.75, 11370}};
  params.rTableHF_ = {{12.5, 16.9, 20.1, 24, 28.6, 34, 40.6, 48.3, 57.6, 68.6, 81.81, 197.5, 116.2, 130}};
  params.shortFibreLength_ = {
      {206, 211.881, 220.382, 235.552, 245.62, 253.909, 255.012, 263.007, 264.348, 268.5, 268.5, 270, 273.5}};
  params.longFibreLength_ = {
      {227.993, 237.122, 241.701, 256.48, 266.754, 275.988, 276.982, 284.989, 286.307, 290.478, 290.5, 292, 295.5}};
  params.attenuationLength_ = {{0.000809654, 0.000713002, 0.000654918, 0.000602767, 0.000566295, 0.000541647,
                                0.000516175, 0.000502512, 0.000504225, 0.000506212, 0.000506275, 0.000487621,
                                0.000473034, 0.000454002, 0.000442383, 0.000441043, 0.00044361,  0.000433124,
                                0.000440188, 0.000435257, 0.000439224, 0.000431385, 0.00041707,  0.000415677,
                                0.000408389, 0.000400293, 0.000400989, 0.000395417, 0.00038936,  0.000383942}};
  params.lambdaLimits_ = {{300, 600}};

  HFFibre fibre(params);
  SECTION("Attenuation") {
    REQUIRE(params.attenuationLength_[0] == fibre.attLength(0));
    REQUIRE(params.attenuationLength_.back() == fibre.attLength(1000));
    const auto binSize = (params.lambdaLimits_[1] - params.lambdaLimits_[0]) / params.attenuationLength_.size();
    for (std::size_t i = 0; i < params.attenuationLength_.size(); ++i) {
      REQUIRE(fibre.attLength(params.lambdaLimits_[0] + binSize * i) == params.attenuationLength_[i]);
    }
  }

  SECTION("zShift") {
    REQUIRE(fibre.zShift({0, 0, 0}, 0) == *(params.longFibreLength_.end() - 1) - 0.5 * params.gParHF_[1]);
  }

  SECTION("tShift") {
    REQUIRE(fibre.zShift({0, 0, 0}, 0) / (params.fractionOfSpeedOfLightInFibre_ * c_light) ==
            fibre.tShift({0, 0, 0}, 0));
  }
}
