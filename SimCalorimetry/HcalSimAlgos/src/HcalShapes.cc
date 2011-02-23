#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloShapeIntegrator.h"
#include "CondFormats/HcalObjects/interface/HcalMCParam.h"
#include "CondFormats/HcalObjects/interface/HcalMCParams.h"
#include "CondFormats/DataRecord/interface/HcalMCParamsRcd.h"


HcalShapes::HcalShapes()
: theMCParams(0),
  theShapes(5),
  theHcalShape(),
  theHFShape(),
  theZDCShape(),
  theSiPMShape()
{
  theShapes[0] = new CaloShapeIntegrator(&theHcalShape);
  theShapes[1] = new CaloShapeIntegrator(&theHcalShape);
  theShapes[2] = new CaloShapeIntegrator(&theSiPMShape);
  theShapes[3] = new CaloShapeIntegrator(&theHFShape);
  theShapes[4] = new CaloShapeIntegrator(&theZDCShape);
}


HcalShapes::~HcalShapes()
{
  for(std::vector<const CaloVShape *>::const_iterator shapeItr = theShapes.begin();
      shapeItr != theShapes.end();  ++shapeItr)
  {
    delete *shapeItr;
  }
  theShapes.clear();
  delete theMCParams;
}


void HcalShapes::beginRun(edm::EventSetup const & es)
{
  edm::ESHandle<HcalMCParams> p;
  es.get<HcalMCParamsRcd>().get(p);
  theMCParams = new HcalMCParams(*p.product()); 
}


void HcalShapes::endRun()
{
  delete theMCParams;
  theMCParams = 0;
}


const CaloVShape * HcalShapes::shape(const DetId & detId) const
{
  if(!theMCParams) {
    return 0;
  }
  int shapeType = theMCParams->getValues(detId)->signalShape();
  return theShapes.at(shapeType);
}

