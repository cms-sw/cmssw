#include "SimCalorimetry/HcalSimAlgos/interface/HcalShapes.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimCalorimetry/CaloSimAlgos/interface/CaloCachedShapeIntegrator.h"
#include "CondFormats/HcalObjects/interface/HcalMCParam.h"
#include "CondFormats/HcalObjects/interface/HcalMCParams.h"
#include "CondFormats/DataRecord/interface/HcalMCParamsRcd.h"
#include "DataFormats/HcalDetId/interface/HcalGenericDetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

HcalShapes::HcalShapes()
: theMCParams(nullptr),
  theTopology(nullptr)
 {
/*
         00 - not used (reserved)
        101 - regular HPD  HB/HE/HO shape
        102 - "special" HB HPD#14 long shape
        201 - SiPMs Zecotec shape   (HO)
        202 - SiPMs Hamamatsu shape (HO)
        203 - SiPMs Hamamatsu shape (HE 2017)
        301 - regular HF PMT shape
        401 - regular ZDC shape
  */

  std::vector<int> theHcalShapeNums = {101,102,103,104,105,123,124,125,201,202,203,205,301};
  // use resize so vector won't invalidate pointers by reallocating memory while filling
  theHcalShapes.resize(theHcalShapeNums.size());
  for(unsigned inum = 0; inum < theHcalShapeNums.size(); ++inum){
    int num = theHcalShapeNums[inum];
    theHcalShapes[inum].setShape(num);
    theShapesPrecise[num] = &theHcalShapes[inum];
    theShapes[num] = new CaloCachedShapeIntegrator(&theHcalShapes[inum]);
  }

  // ZDC not yet defined in CalibCalorimetry/HcalAlgos/src/HcalPulseShapes.cc
  theShapesPrecise[ZDC] = &theZDCShape;
  theShapes[ZDC] = new CaloCachedShapeIntegrator(&theZDCShape);
}


HcalShapes::~HcalShapes()
{
  for(auto& shapeItr : theShapes)
  {
    delete shapeItr.second;
  }
  theShapes.clear();
  if (theMCParams!=0) delete theMCParams;
  if (theTopology!=0) delete theTopology;
}


void HcalShapes::beginRun(edm::EventSetup const & es)
{
  edm::ESHandle<HcalMCParams> p;
  es.get<HcalMCParamsRcd>().get(p);
  theMCParams = new HcalMCParams(*p.product()); 

// here we are making a _copy_ so we need to add a copy of the topology...
  
  edm::ESHandle<HcalTopology> htopo;
  es.get<HcalRecNumberingRecord>().get(htopo);
  theTopology=new HcalTopology(*htopo);
  theMCParams->setTopo(theTopology);
}


void HcalShapes::endRun()
{
  if (theMCParams) delete theMCParams;
  theMCParams = 0;
  if (theTopology) delete theTopology;
  theTopology = 0;
}


const CaloVShape * HcalShapes::shape(const DetId & detId, bool precise) const
{
  if(!theMCParams) {
    return defaultShape(detId);
  }
  int shapeType = theMCParams->getValues(detId)->signalShape();
  const auto& myShapes = getShapeMap(precise);
  auto shapeMapItr = myShapes.find(shapeType);
  if(shapeMapItr == myShapes.end()) {
       edm::LogWarning("HcalShapes") << "HcalShapes::shape - shapeType ?  = "
				     << shapeType << std::endl;
    return defaultShape(detId,precise);
  } else {
    return shapeMapItr->second;
  }
}

const CaloVShape * HcalShapes::defaultShape(const DetId & detId, bool precise) const
{
  // try to figure the appropriate shape
  const CaloVShape * result;
  const auto& myShapes = getShapeMap(precise);
  HcalGenericDetId::HcalGenericSubdetector subdet 
    = HcalGenericDetId(detId).genericSubdet();
  if(subdet == HcalGenericDetId::HcalGenBarrel 
  || subdet == HcalGenericDetId::HcalGenEndcap) result = myShapes.find(HPD)->second;
  else if(subdet == HcalGenericDetId::HcalGenOuter) result = myShapes.find(HPD)->second;
  else if(subdet == HcalGenericDetId::HcalGenForward) result = myShapes.find(HF)->second;
  else if(subdet == HcalGenericDetId::HcalGenZDC) result = myShapes.find(ZDC)->second;
  else result = 0;

  edm::LogWarning("HcalShapes") << "Cannot find HCAL MC Params, so the default one is taken for subdet " << subdet;  

  return result;
}

const HcalShapes::ShapeMap& HcalShapes::getShapeMap(bool precise) const {
  return precise ? theShapesPrecise : theShapes;
}
