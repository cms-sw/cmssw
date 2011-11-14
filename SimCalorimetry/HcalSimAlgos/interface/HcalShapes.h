#ifndef HcalSimAlgos_HcalShapes_h
#define HcalSimAlgos_HcalShapes_h

/** A class which decides which shape to return,
   based on the DetId
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalLVShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HFShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalSiPMShape.h"
#include <map>
class CaloVShape;
class DetId;
class HcalMCParams;

class HcalShapes : public CaloShapes
{
public:
  enum {HPD=101, LONG=102, ZECOTEK=201, HAMAMATSU=202, HF=301, ZDC=401};
  HcalShapes();
  ~HcalShapes();

  void beginRun(edm::EventSetup const & es);
  void endRun();

  virtual const CaloVShape * shape(const DetId & detId) const;

private:
  // hardcoded, if we can't figure it out form the DB
  const CaloVShape * defaultShape(const DetId & detId) const;
  const HcalMCParams * theMCParams;
  typedef std::map<int, const CaloVShape *> ShapeMap;
  ShapeMap theShapes;
  HcalShape theHcalShape;
  HcalLVShape theHcalLVShape;
  HFShape theHFShape;
  ZDCShape theZDCShape;
  HcalSiPMShape theSiPMShape;
};

#endif

