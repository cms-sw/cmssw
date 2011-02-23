#ifndef HcalSimAlgos_HcalShapes_h
#define HcalSimAlgos_HcalShapes_h

/** A class which decides which shape to return,
   based on the DetId
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"
#include <vector>
class CaloVShape;
class DetId;
class HcalMCParams;

class HcalShapes : public CaloShapes
{
public:
  HcalShapes();
  ~HcalShapes();

  void beginRun(edm::EventSetup const & es);
  void endRun();

  virtual const CaloVShape * shape(const DetId & detId) const;

private:
  const HcalMCParams * theMCParams;
  std::vector<const CaloVShape *> theShapes;
  HcalShape theHcalShape;
  HFShape theHFShape;
  ZDCShape theZDCShape;
  HcalSiPMShape theSiPMShape;
};

#endif

