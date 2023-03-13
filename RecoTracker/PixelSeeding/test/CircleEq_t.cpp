#include "RecoTracker/PixelSeeding/interface/CircleEq.h"
#include <cassert>

struct OriCircle {
  using T = float;

  float radius = 0;
  float x_center = 0;
  float y_center = 0;

  constexpr OriCircle(T x1, T y1, T x2, T y2, T x3, T y3) { compute(x1, y1, x2, y2, x3, y3); }

  // dca to origin
  constexpr T dca0() const { return std::sqrt(x_center * x_center + y_center * y_center) - radius; }

  // dca to given point
  constexpr T dca(T x, T y) const {
    x -= x_center;
    y -= y_center;
    return std::sqrt(x * x + y * y) - radius;
  }

  constexpr void compute(T x1, T y1, T x2, T y2, T x3, T y3) {
    auto det = (x1 - x2) * (y2 - y3) - (x2 - x3) * (y1 - y2);

    auto offset = x2 * x2 + y2 * y2;

    auto bc = (x1 * x1 + y1 * y1 - offset) * 0.5f;

    auto cd = (offset - x3 * x3 - y3 * y3) * 0.5f;

    auto idet = 1.f / det;

    x_center = (bc * (y2 - y3) - cd * (y1 - y2)) * idet;
    y_center = (cd * (x1 - x2) - bc * (x2 - x3)) * idet;

    radius = std::sqrt((x2 - x_center) * (x2 - x_center) + (y2 - y_center) * (y2 - y_center));
  }
};

#include <iostream>

template <typename T>
bool equal(T a, T b) {
  //  return float(a-b)==0;
  return std::abs(float(a - b)) < std::abs(0.01f * a);
}

int main() {
  float r1 = 4, r2 = 8, r3 = 15;
  for (float phi = -3; phi < 3.1; phi += 0.5) {
    float x1 = r1 * cos(phi);
    float x2 = r2 * cos(phi);
    float y1 = r1 * sin(phi);
    float y2 = r2 * sin(phi);
    for (float phi3 = phi - 0.31; phi3 < phi + 0.31; phi3 += 0.05) {
      float x3 = r3 * cos(phi3);
      float y3 = r3 * sin(phi3);

      OriCircle ori(x1, y1, x2, y2, x3, y3);
      CircleEq<float> eq(x1, y1, x2, y2, x3, y3);
      // std::cout << "r " << ori.radius <<' '<< eq.radius() << std::endl;
      assert(equal(ori.radius, std::abs(eq.radius())));
      auto c = eq.center();
      auto dir = eq.cosdir();
      assert(equal(1.f, dir.first * dir.first + dir.second * dir.second));
      assert(equal(ori.x_center, c.first));
      assert(equal(ori.y_center, c.second));
      // std::cout << "dca " << ori.dca0() <<' '<< eq.radius()*eq.dca0() << std::endl;
      assert(equal(std::abs(ori.dca0()), std::abs(eq.radius() * eq.dca0())));
      // std::cout << "dca " << ori.dca(1.,1.) <<' '<< eq.radius()*eq.dca(1.,1.) << std::endl;
      assert(equal(std::abs(ori.dca(1., 1.)), std::abs(eq.radius() * eq.dca(1., 1.))));
    }
  }

  return 0;
}
