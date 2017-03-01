#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFinegrainBit.h"

#include <cassert>

std::bitset<4>
HcalFinegrainBit::compute(const HcalFinegrainBit::Tower& tower) const
{
   if (version_ == 0) {
      // Currently assumes that the bits that are set are mutually
      // exclusive!
      assert((tower[is_mip] & tower[is_above_mip]).count() == 0);

      std::bitset<4> result;

      // First layer consistent with a MIP
      result[0] = tower[is_mip][0];

      // First layer consistent with a MIP, at least one layer with more
      // than MIP energy deposition
      result[1] = result[0] & (tower[is_above_mip].count() > 0);

      // There layers consistent with a MIP
      result[2] = tower[is_mip].count() >= 3;

      // Unset
      result[3] = false;

      return result;
   }
   return 0;
}
