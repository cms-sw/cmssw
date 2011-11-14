#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloCachedShapeIntegrator.h"
#include "CondFormats/HcalObjects/interface/HcalMCParam.h"
#include "CondFormats/HcalObjects/interface/HcalMCParams.h"
#include "CondFormats/DataRecord/interface/HcalMCParamsRcd.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HcalShapes::HcalShapes()
: theMCParams(0),
  theShapes(),
  theHcalShape(),
  theHcalLVShape(),
  theHFShape(),
  theZDCShape(),
  theSiPMShape()
{
/*
         00 - not used (reserved)
        101 - regular HPD  HB/HE/HO shape
        102 - "special" HB HPD#14 long shape
        201 - SiPMs Zecotec shape   (HO)
        202 - SiPMs Hamamatsu shape (HO)
        301 - regular HF PMT shape
        401 - regular ZDC shape
  */
  theShapes[HPD] = new CaloCachedShapeIntegrator(&theHcalShape);
  theShapes[LONG] = new CaloCachedShapeIntegrator(&theHcalLVShape);
  theShapes[ZECOTEK] = new CaloCachedShapeIntegrator(&theSiPMShape);
  theShapes[HAMAMATSU] = theShapes[ZECOTEK];
  theShapes[HF] = new CaloCachedShapeIntegrator(&theHFShape);
  theShapes[ZDC] = new CaloCachedShapeIntegrator(&theZDCShape);

  // backward-compatibility with old scheme
  theShapes[0] = theShapes[HPD];
  //FIXME "special" HB
  theShapes[1] = theShapes[LONG];
  theShapes[2] = theShapes[ZECOTEK];
  theShapes[3] = theShapes[HF];
  theShapes[4] = theShapes[ZDC];
}


HcalShapes::~HcalShapes()
{
  for(ShapeMap::const_iterator shapeItr = theShapes.begin();
      shapeItr != theShapes.end();  ++shapeItr)
  {
    delete shapeItr->second;
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
    return defaultShape(detId);
  }
  int shapeType = theMCParams->getValues(detId)->signalShape();
  ShapeMap::const_iterator shapeMapItr = theShapes.find(shapeType);
  if(shapeMapItr == theShapes.end()) {
    return defaultShape(detId);
  } else {
    return shapeMapItr->second;
  }
}

const CaloVShape * HcalShapes::defaultShape(const DetId & detId) const
{
  // try to figure the appropriate shape
  const CaloVShape * result;
  HcalGenericDetId::HcalGenericSubdetector subdet 
    = HcalGenericDetId(detId).genericSubdet();
  if(subdet == HcalGenericDetId::HcalGenBarrel 
  || subdet == HcalGenericDetId::HcalGenEndcap) result = theShapes.find(0)->second;
  else if(subdet == HcalGenericDetId::HcalGenOuter) result = theShapes.find(2)->second;
  else if(subdet == HcalGenericDetId::HcalGenForward) result = theShapes.find(3)->second;
  else if(subdet == HcalGenericDetId::HcalGenZDC) result = theShapes.find(3)->second;
  else result = 0;

  edm::LogWarning("HcalShapes") << "Cannot find HCAL MC Params, so the defalut one is taken for  subdet " << subdet;  

  return result;
}
