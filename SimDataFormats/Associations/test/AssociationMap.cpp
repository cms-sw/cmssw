#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "catch2/catch_all.hpp"

#include <sstream>
#include <string>

using namespace ticl;
using Catch::Approx;

TEST_CASE("TICLAssociationMap - mapWithFraction basic insert/sort/accumulate", "[TICLAssociationMap]") {
  TICLAssociationMap<mapWithFraction> m(3);

  REQUIRE(m.size() == 3);

  // insert three associations for key 0
  m.insert(0, 2u, 0.40f);
  m.insert(0, 1u, 0.70f);
  m.insert(0, 3u, 0.55f);

  REQUIRE(m[0].size() == 3);

  // sort by fraction (default)
  m.sort(false);
  REQUIRE(m[0][0].index() == 1u);
  REQUIRE(m[0][1].index() == 3u);
  REQUIRE(m[0][2].index() == 2u);

  // fractions are as inserted (and accessible)
  REQUIRE(m[0][0].fraction() == Approx(0.70f));
  REQUIRE(m[0][1].fraction() == Approx(0.55f));
  REQUIRE(m[0][2].fraction() == Approx(0.40f));

  m.insert(0, 1u, 0.10f);
  // find element with index==1 and check fraction increased
  auto it = std::find_if(m[0].begin(), m[0].end(), [](auto const& e) { return e.index() == 1u; });
  REQUIRE(it != m[0].end());
  REQUIRE(it->fraction() == Approx(0.80f));
}

TEST_CASE("TICLAssociationMap - mapWithFractionAndScore sort by value and by score", "[TICLAssociationMap]") {
  TICLAssociationMap<mapWithFractionAndScore> m(2);

  // insert several with scores
  m.insert(0, 2u, 0.40f, 0.20f);
  m.insert(0, 1u, 0.70f, 0.50f);
  m.insert(0, 3u, 0.55f, 0.90f);

  // default sort (by fraction)
  m.sort(false);
  REQUIRE(m[0][0].index() == 1u);
  REQUIRE(m[0][1].index() == 3u);
  REQUIRE(m[0][2].index() == 2u);

  REQUIRE(m[0][0].fraction() == Approx(0.70f));
  REQUIRE(m[0][0].score() == Approx(0.50f));

  // sort by score
  m.sort(true);
  REQUIRE(m[0][0].index() == 3u);  // highest score
  REQUIRE(m[0][0].score() == Approx(0.90f));
  REQUIRE(m[0][1].score() == Approx(0.50f));
  REQUIRE(m[0][2].score() == Approx(0.20f));
}

TEST_CASE("TICLAssociationMap - mapWithSharedEnergy basic insert/sort", "[TICLAssociationMap]") {
  TICLAssociationMap<mapWithSharedEnergy> m(2);

  m.insert(0, 5u, 1.5f);
  m.insert(0, 2u, 3.0f);
  m.insert(0, 9u, 2.2f);

  m.sort(false);
  REQUIRE(m[0][0].index() == 2u);
  REQUIRE(m[0][1].index() == 9u);
  REQUIRE(m[0][2].index() == 5u);

  REQUIRE(m[0][0].sharedEnergy() == Approx(3.0f));
}

TEST_CASE("TICLAssociationMap - one-to-one map behavior", "[TICLAssociationMap]") {
  TICLAssociationMap<oneToOneMapWithFraction> m(3);

  // insert entries for one-to-one map
  m.insert(0u, 2u, 0.40f);
  m.insert(1u, 0u, 0.70f);
  m.insert(2u, 1u, 0.20f);

  // sort is no-op for one-to-one (should compile & do nothing)
  m.sort(false);

  // operator[] returns AssociationElement (not a vector)
  REQUIRE(m[0].index() == 2u);
  REQUIRE(m[1].index() == 0u);
  REQUIRE(m[2].index() == 1u);

  REQUIRE(m[0].fraction() == Approx(0.40f));
  REQUIRE(m[1].fraction() == Approx(0.70f));
  REQUIRE(m[2].fraction() == Approx(0.20f));

  // at() should return the element when valid
  auto const& e0 = m.at(0u);
  REQUIRE(e0.index() == 2u);
  REQUIRE(e0.isValid());

  // default-constructed one-to-one element should be invalid: check throw
  TICLAssociationMap<oneToOneMapWithFraction> m2(1);
  REQUIRE_THROWS_AS(m2.at(0u), std::out_of_range);
}

TEST_CASE("TICLAssociationMap - print output contains expected text", "[TICLAssociationMap]") {
  TICLAssociationMap<mapWithFraction> m(2);
  m.insert(0, 1u, 0.5f);
  m.insert(1, 2u, 0.6f);

  std::ostringstream os;
  m.print(os);
  const auto s = os.str();

  REQUIRE(!s.empty());
  // should contain Index 0/1 and the word "Fraction"
  REQUIRE(s.find("Index 0") != std::string::npos);
  REQUIRE(s.find("Index 1") != std::string::npos);
  REQUIRE(s.find("Fraction") != std::string::npos);
}
