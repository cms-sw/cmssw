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
: theMCParams(0),
  theShapes(),
  theZDCShape(),
  theHcalShape101(),
  theHcalShape102(),
  theHcalShape103(),
  theHcalShape104(),
  theHcalShape105(),
  theHcalShape123(),
  theHcalShape124(),
  theHcalShape125(),
  theHcalShape201(),
  theHcalShape202(),
  theHcalShape301(),
  theHcalShape401()
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

/*  
  theShapes[HPD] = new CaloCachedShapeIntegrator(&theHcalShape);
  theShapes[LONG] = theShapes[HPD];
  theShapes[ZECOTEK] = new CaloCachedShapeIntegrator(&theSiPMShape);
  theShapes[HAMAMATSU] = theShapes[ZECOTEK];
  theShapes[HF] = new CaloCachedShapeIntegrator(&theHFShape);
*/

  theHcalShape101.setShape(101); 
  theShapes[101] = new CaloCachedShapeIntegrator(&theHcalShape101);
  theHcalShape102.setShape(102);                  
  theShapes[102] = new CaloCachedShapeIntegrator(&theHcalShape102);
  theHcalShape103.setShape(103);                  
  theShapes[103] = new CaloCachedShapeIntegrator(&theHcalShape103);
  theHcalShape104.setShape(104);                  
  theShapes[104] = new CaloCachedShapeIntegrator(&theHcalShape104);
  theHcalShape104.setShape(105);
  theShapes[105] = new CaloCachedShapeIntegrator(&theHcalShape105); // HPD new 
  theHcalShape123.setShape(123);                  
  theShapes[123] = new CaloCachedShapeIntegrator(&theHcalShape123);
  theHcalShape124.setShape(124);                  
  theShapes[124] = new CaloCachedShapeIntegrator(&theHcalShape124);
  theHcalShape125.setShape(125);
  theShapes[125] = new CaloCachedShapeIntegrator(&theHcalShape125);
  theHcalShape201.setShape(201);                  
  theShapes[201] = new CaloCachedShapeIntegrator(&theHcalShape201);
  theHcalShape202.setShape(202);                  
  theShapes[202] = new CaloCachedShapeIntegrator(&theHcalShape202);
  theHcalShape301.setShape(301);
  theShapes[301] = new CaloCachedShapeIntegrator(&theHcalShape301);
  //    ZDC not yet defined in CalibCalorimetry/HcalAlgos/src/HcalPulseShapes.cc
  // theHcalShape401(401);
  // theShapes[401] = new CaloCachedShapeIntegrator(&theHcalShape401);
  theShapes[ZDC] = new CaloCachedShapeIntegrator(&theZDCShape);



  // backward-compatibility with old scheme

  theShapes[0] = theShapes[HPD];
  //FIXME "special" HB
  theShapes[1] = theShapes[LONG];
  theShapes[2] = theShapes[ZECOTEK];
  theShapes[3] = theShapes[HF];
  theShapes[4] = theShapes[ZDC];

  theMCParams=0;
  theTopology=0;
}


HcalShapes::~HcalShapes()
{
  for(ShapeMap::const_iterator shapeItr = theShapes.begin();
      shapeItr != theShapes.end();  ++shapeItr)
  {
    delete shapeItr->second;
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


const CaloVShape * HcalShapes::shape(const DetId & detId) const
{
  if(!theMCParams) {
    return defaultShape(detId);
  }
  int shapeType = theMCParams->getValues(detId)->signalShape();
  /*
	  HcalDetId cell(detId);
	  int sub     = cell.subdet();
	  int depth   = cell.depth();
	  int inteta  = cell.ieta();
	  int intphi  = cell.iphi();
	  
	  std::cout << "(SIM)HcalShapes::shape  cell:" 
		    << " sub, ieta, iphi, depth = " 
		    << sub << "  " << inteta << "  " << intphi 
		    << "  " << depth  << " => ShapeId "<<  shapeType 
		    << std::endl;
  */
  ShapeMap::const_iterator shapeMapItr = theShapes.find(shapeType);
  if(shapeMapItr == theShapes.end()) {
       edm::LogWarning("HcalShapes") << "HcalShapes::shape - shapeType ?  = "
				     << shapeType << std::endl;
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
