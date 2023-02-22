#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimCalorimetry/EcalSimAlgos/interface/ComponentShapeCollection.h"

// #define component_shape_debug 1
void ComponentShapeCollection::setEventSetup(const edm::EventSetup& evtSetup) {
#ifdef component_shape_debug
  std::cout << "ComponentShapeCollection::setEventSetup called " << std::endl;
#endif
  buildMe(&evtSetup);
  for (int i = 0; i < m_nDepthBins; ++i) {
    m_shapeArr[i]->setEventSetup(evtSetup, false);
  }
}

void ComponentShapeCollection::buildMe(const edm::EventSetup* evtSetup) {
#ifdef component_shape_debug
  std::cout << "ComponentShapeCollection::buildMe called " << std::endl;
#endif
  fillCollection(m_useDBShape);
};

void ComponentShapeCollection::fillCollection(edm::ConsumesCollector iC) {
#ifdef component_shape_debug
  std::cout << "ComponentShapeCollection::fillCollection(edm::ConsumesCollector iC) called " << std::endl;
#endif
  //m_shapeArr->clear();
  for (int i = 0; i < m_nDepthBins; ++i) {
    m_shapeArr[i] = std::make_shared<ComponentShape>(i, espsToken_);
  }
}

void ComponentShapeCollection::fillCollection(bool useDBShape = false) {
#ifdef component_shape_debug
  std::cout << "ComponentShapeCollection::fillCollection(bool useDBShape) called " << std::endl;
#endif
  //m_shapeArr->clear();
  if (useDBShape) {
    for (int i = 0; i < m_nDepthBins; ++i) {
      m_shapeArr[i] = std::make_shared<ComponentShape>(i, espsToken_);
    }
  } else {
    for (int i = 0; i < m_nDepthBins; ++i) {
      m_shapeArr[i] = std::make_shared<ComponentShape>(i);
    }
  }
}

const std::shared_ptr<ComponentShape> ComponentShapeCollection::at(int depthIndex) const {
  if (0 > toDepthBin(depthIndex) || toDepthBin(depthIndex) > m_nDepthBins - 1)
    throw cms::Exception("ComponentShape:: invalid depth requested");
  return m_shapeArr[toDepthBin(depthIndex)];
}

int ComponentShapeCollection::toDepthBin(int index) { return index >> 3; }

int ComponentShapeCollection::maxDepthBin() { return m_nDepthBins - 1; }
