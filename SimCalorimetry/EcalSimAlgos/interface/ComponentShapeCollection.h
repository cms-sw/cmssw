#ifndef SimCalorimetry_EcalSimAlgos_ComponentShapeCollection_h
#define SimCalorimetry_EcalSimAlgos_ComponentShapeCollection_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ComponentShape.h"

class ComponentShapeCollection {
public:
  ComponentShapeCollection(bool useDBShape, edm::ConsumesCollector iC)
      : m_useDBShape(useDBShape), m_thresh(0.0), espsToken_(iC.esConsumes()) {
    fillCollection(iC);
  }
  ComponentShapeCollection(edm::ConsumesCollector iC) : ComponentShapeCollection(true, iC) {}
  ComponentShapeCollection(bool useDBShape) : m_useDBShape(useDBShape), m_thresh(0.0) { fillCollection(useDBShape); }

  ~ComponentShapeCollection() {}

  void setEventSetup(const edm::EventSetup& evtSetup);

  const std::shared_ptr<ComponentShape> at(int depthIndex) const;
  static int toDepthBin(int index);
  static int maxDepthBin();

protected:
  void buildMe(const edm::EventSetup* es = nullptr);
  void fillCollection(bool useDBShape);
  void fillCollection(edm::ConsumesCollector iC);

  bool m_useDBShape;
  double m_thresh;

private:
  const static int m_nDepthBins = 23;  // dictated by SimG4CMS/Calo/src/ECalSD.cc, 230 mm / 10 mm
  edm::ESGetToken<EcalSimComponentShape, EcalSimComponentShapeRcd> espsToken_;
  std::shared_ptr<ComponentShape> m_shapeArr[m_nDepthBins];
};

#endif
