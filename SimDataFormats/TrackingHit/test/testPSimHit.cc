#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <iostream>
#include <iomanip>
#include <cassert>

int main() {
  Local3DPoint dummy(0, 0, 0);

  for (int procId = 1; procId < 404; procId++) {
    for (unsigned short itype = 0; itype < 8; itype++) {
      PSimHit testHit(dummy, dummy, 0., 0., 0., 11, 0, 0, 0., 0., procId);
      testHit.setHitProdType(itype);
      std::cout << " hit procId = " << std::fixed << std::setw(8) << procId << " sim procId = " << std::setw(8)
                << testHit.processType() << " hit type = " << std::setw(8) << itype << " sim type = " << std::setw(8)
                << testHit.hitProdType() << std::endl;

      assert(procId == testHit.processType());
      assert(itype == testHit.hitProdType());
    }
  }

  return 0;
}
