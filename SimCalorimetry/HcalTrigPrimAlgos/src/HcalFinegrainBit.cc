#include "SimCalorimetry/HcalTrigPrimAlgos/interface/HcalFinegrainBit.h"

#include <cassert>

std::bitset<4>
HcalFinegrainBit::compute(const HcalFinegrainBit::Tower& tower) const
{
   if (version_ == 0) {
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
   if (version_ == 1) {
      std::bitset<4> result;

      // All algorithms the same for testing purposes
      result[0] = result[1] = result[2] = result [3] = tower[is_mip][0];

      return result;
   }
   if (version_ == 2) {
      std::bitset<4> result;

      // All algorithms the same for testing purposes
      result[0] = result[1] = result[2] = result [3] = true;

      return result;
   }
   return 0;
}
