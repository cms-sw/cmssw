#ifndef MU_END_STRIP_HIT_SIM_H
#define MU_END_STRIP_HIT_SIM_H

/** \class CSCStripHitSim
 *
 * Class which builds simulated strip hits from wire
 * hits during digitization of Endcap Muon CSCs.
 *
 * \author Rick Wilkinson
 *
 */

#include "SimMuon/CSCDigitizer/src/CSCGattiFunction.h"
#include <vector>
#include "SimMuon/CSCDigitizer/src/CSCDetectorHit.h"

// declarations
class CSCLayer;
//class CSCDetectorHit;

class CSCStripHitSim
{
public:
  // make strip hits from the given wire hits
  std::vector<CSCDetectorHit> & simulate(const CSCLayer * layer, 
                       const std::vector<CSCDetectorHit> & wireHits);
private:
  CSCGattiFunction theGattiFunction;
  std::vector<CSCDetectorHit> newStripHits;
};

#endif
