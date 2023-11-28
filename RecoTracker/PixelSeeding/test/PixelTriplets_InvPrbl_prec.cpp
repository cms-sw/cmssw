#include "RecoTracker/PixelSeeding/plugins/ThirdHitPredictionFromInvParabola.cc"

#include <iostream>
#include <string>

namespace test {
  namespace PixelTriplets_InvPrbl_prec {
    int test() {
      std::string c("++Constr");
      std::string r("++R");
      std::string a;
      double par[7];
      double q[2];
      while (std::cin) {
        ThirdHitPredictionFromInvParabola pred;
        std::cin >> a;
        if (a == c) {
          for (auto& p : par)
            std::cin >> p;
          pred = ThirdHitPredictionFromInvParabola(par[0], par[1], par[2], par[3], par[4], par[5], par[6]);
          std::cout << "ip min, max " << pred.theIpRangePlus.min() << " " << pred.theIpRangePlus.max() << "  "
                    << pred.theIpRangeMinus.min() << " " << pred.theIpRangeMinus.max() << std::endl;
        } else if (a == r) {
          std::cin >> q[0] >> q[1];
          {
            auto rp = pred.rangeRPhi(q[0], 1);
            auto rn = pred.rangeRPhi(q[0], -1);
            std::cout << "range " << rp.min() << " " << rp.max() << " " << rn.min() << " " << rn.max() << std::endl;
          }
          {
            auto rp = pred.rangeRPhi(q[1], 1);
            auto rn = pred.rangeRPhi(q[1], -1);
            std::cout << "range " << rp.min() << " " << rp.max() << " " << rn.min() << " " << rn.max() << std::endl;
          }
        }
      }
      return 0;
    }
  }  // namespace PixelTriplets_InvPrbl_prec
}  // namespace test

int main() { return test::PixelTriplets_InvPrbl_prec::test(); }
