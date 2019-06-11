#ifndef HcalSimAlgos_HcalShapes_h
#define HcalSimAlgos_HcalShapes_h

/** A class which decides which shape to return,
   based on the DetId
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapes.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HcalShape.h"
#include "SimCalorimetry/HcalSimAlgos/interface/ZDCShape.h"
#include "CalibFormats/HcalObjects/interface/HcalDbService.h"
#include <vector>
#include <map>
class CaloVShape;
class DetId;

class HcalShapes : public CaloShapes {
public:
  enum { HPD = 101, LONG = 102, ZECOTEK = 201, HAMAMATSU = 202, HE2017 = 203, HE2018 = 206, HF = 301, ZDC = 401 };
  HcalShapes();
  ~HcalShapes() override;

  void setDbService(const HcalDbService* service) { theDbService = service; }

  const CaloVShape* shape(const DetId& detId, bool precise = false) const override;

private:
  typedef std::map<int, const CaloVShape*> ShapeMap;
  // hardcoded, if we can't figure it out from the DB
  const CaloVShape* defaultShape(const DetId& detId, bool precise = false) const;
  const ShapeMap& getShapeMap(bool precise) const;
  const HcalDbService* theDbService;
  ShapeMap theShapes;
  ShapeMap theShapesPrecise;
  ZDCShape theZDCShape;
  //   list of vShapes.
  std::vector<HcalShape> theHcalShapes;
};

#endif
