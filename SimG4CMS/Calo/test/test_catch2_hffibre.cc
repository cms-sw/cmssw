#include "catch.hpp"
#include "SimG4CMS/Calo/interface/HFFibre.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

namespace test_hffibre {
  HFFibre::Params defaultParams() {
    HFFibre::Params fibreParams;
    //Taken from values used by IB workflow 250202.181
    fibreParams.fractionOfSpeedOfLightInFibre_ = 0.5;
    fibreParams.gParHF_ = {{220, 1650, 300, 0, 11150, 3.75, 11370}};
    fibreParams.rTableHF_ = {{12.5 * cm,
                              16.9 * cm,
                              20.1 * cm,
                              24 * cm,
                              28.6 * cm,
                              34 * cm,
                              40.6 * cm,
                              48.3 * cm,
                              57.6 * cm,
                              68.6 * cm,
                              81.81 * cm,
                              197.5 * cm,
                              116.2 * cm,
                              130 * cm}};
    fibreParams.shortFibreLength_ = {{206 * cm,
                                      211.881 * cm,
                                      220.382 * cm,
                                      235.552 * cm,
                                      245.62 * cm,
                                      253.909 * cm,
                                      255.012 * cm,
                                      263.007 * cm,
                                      264.348 * cm,
                                      268.5 * cm,
                                      268.5 * cm,
                                      270 * cm,
                                      273.5 * cm}};
    fibreParams.longFibreLength_ = {{227.993 * cm,
                                     237.122 * cm,
                                     241.701 * cm,
                                     256.48 * cm,
                                     266.754 * cm,
                                     275.988 * cm,
                                     276.982 * cm,
                                     284.989 * cm,
                                     286.307 * cm,
                                     290.478 * cm,
                                     290.5 * cm,
                                     292 * cm,
                                     295.5 * cm}};
    fibreParams.attenuationLength_ = {
        {0.000809654 / cm, 0.000713002 / cm, 0.000654918 / cm, 0.000602767 / cm, 0.000566295 / cm, 0.000541647 / cm,
         0.000516175 / cm, 0.000502512 / cm, 0.000504225 / cm, 0.000506212 / cm, 0.000506275 / cm, 0.000487621 / cm,
         0.000473034 / cm, 0.000454002 / cm, 0.000442383 / cm, 0.000441043 / cm, 0.00044361 / cm,  0.000433124 / cm,
         0.000440188 / cm, 0.000435257 / cm, 0.000439224 / cm, 0.000431385 / cm, 0.00041707 / cm,  0.000415677 / cm,
         0.000408389 / cm, 0.000400293 / cm, 0.000400989 / cm, 0.000395417 / cm, 0.00038936 / cm,  0.000383942 / cm}};
    fibreParams.lambdaLimits_ = {{300, 600}};
    return fibreParams;
  }
}  // namespace test_hffibre

TEST_CASE("test HFFibre", "[HFFibre]") {
  HFFibre::Params params = test_hffibre::defaultParams();

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
