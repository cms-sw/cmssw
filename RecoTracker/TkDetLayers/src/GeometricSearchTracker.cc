#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

GeometricSearchTracker::GeometricSearchTracker(const vector<BarrelDetLayer*>& pxlBar,
					       const vector<BarrelDetLayer*>& tib,
					       const vector<BarrelDetLayer*>& tob,
					       const vector<ForwardDetLayer*>& negPxlFwd,
					       const vector<ForwardDetLayer*>& negTid,
					       const vector<ForwardDetLayer*>& negTec,
					       const vector<ForwardDetLayer*>& posPxlFwd,
					       const vector<ForwardDetLayer*>& posTid,
					       const vector<ForwardDetLayer*>& posTec):
  thePixelBarrelLayers(pxlBar.begin(),pxlBar.end()),
  theTibLayers(tib.begin(),tib.end()),
  theTobLayers(tob.begin(),tob.end()),
  theNegPixelForwardLayers(negPxlFwd.begin(),negPxlFwd.end()),
  theNegTidLayers(negTid.begin(),negTid.end()),
  theNegTecLayers(negTec.begin(),negTec.end()),
  thePosPixelForwardLayers(posPxlFwd.begin(),posPxlFwd.end()),
  thePosTidLayers(posTid.begin(),posTid.end()),
  thePosTecLayers(posTec.begin(),posTec.end())
{
  theBarrelLayers.assign(thePixelBarrelLayers.begin(),thePixelBarrelLayers.end());
  theBarrelLayers.insert(theBarrelLayers.end(),theTibLayers.begin(),theTibLayers.end());
  theBarrelLayers.insert(theBarrelLayers.end(),theTobLayers.begin(),theTobLayers.end());

  theNegForwardLayers.assign(theNegPixelForwardLayers.begin(),theNegPixelForwardLayers.end());
  theNegForwardLayers.insert(theNegForwardLayers.end(),theNegTidLayers.begin(),theNegTidLayers.end());
  theNegForwardLayers.insert(theNegForwardLayers.end(),theNegTecLayers.begin(),theNegTecLayers.end());

  thePosForwardLayers.assign(thePosPixelForwardLayers.begin(),thePosPixelForwardLayers.end());
  thePosForwardLayers.insert(thePosForwardLayers.end(),thePosTidLayers.begin(),thePosTidLayers.end());
  thePosForwardLayers.insert(thePosForwardLayers.end(),thePosTecLayers.begin(),thePosTecLayers.end());


  theForwardLayers.assign(theNegForwardLayers.begin(),theNegForwardLayers.end());
  theForwardLayers.insert(theForwardLayers.end(),thePosForwardLayers.begin(),thePosForwardLayers.end());
  theAllLayers.assign(theBarrelLayers.begin(),theBarrelLayers.end());
  theAllLayers.insert(theAllLayers.end(),
		      theForwardLayers.begin(),
		      theForwardLayers.end());

  // number the layers 
  int sq=0;
  for (auto l : theAllLayers) 
    (*l).setSeqNum(sq++);

  edm::LogInfo("TkDetLayers")
    << "------ GeometricSearchTracker constructed with: ------" << "\n"
    << "n pxlBarLayers: " << this->pixelBarrelLayers().size() << "\n"
    << "n tibLayers:    " << this->tibLayers().size() << "\n"
    << "n tobLayers:    " << this->tobLayers().size() << "\n"
    << "n negPxlFwdLayers: " << this->negPixelForwardLayers().size() << "\n"
    << "n posPxlFwdLayers: " << this->posPixelForwardLayers().size() << "\n"
    << "n negTidLayers: " << this->negTidLayers().size() << "\n"
    << "n posTidLayers: " << this->posTidLayers().size() << "\n"
    << "n negTecLayers: " << this->negTecLayers().size() << "\n"
    << "n posTecLayers: " << this->posTecLayers().size() << "\n"
    
    << "n barreLayers:  " << this->barrelLayers().size() << "\n"
    << "n negforwardLayers: " << this->negForwardLayers().size() << "\n"
    << "n posForwardLayers: " << this->posForwardLayers().size() 
    << "\nn Total :     "     << theAllLayers.size() << " " << sq
    << std::endl;

    for (auto l : theAllLayers)
      edm::LogInfo("TkDetLayers") << (*l).seqNum()<< ": " << (*l).subDetector() << ", ";
    edm::LogInfo("TkDetLayers") << std::endl;

}


GeometricSearchTracker::~GeometricSearchTracker(){
  for(vector<DetLayer*>::const_iterator it=theAllLayers.begin(); it!=theAllLayers.end();it++){
    delete *it;
  }
  
}


#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

const DetLayer*          
GeometricSearchTracker::idToLayer(const DetId& id) const
{
  switch(id.subdetId()) {
  case StripSubdetector::TIB:
    //edm::LogInfo(TkDetLayers) << "TIB layer n: " << TIBDetId(id).layer() ;
    return theTibLayers[TIBDetId(id).layer()-1];
    break;

  case StripSubdetector::TOB:
    //edm::LogInfo(TkDetLayers) << "TOB layer n: " << TOBDetId(id).layer() ;
    return theTobLayers[TOBDetId(id).layer()-1];
    break;

  case StripSubdetector::TID:
    //edm::LogInfo(TkDetLayers) << "TID wheel n: " << TIDDetId(id).wheel() ;
    if(TIDDetId(id).side() ==1 ) {
      return theNegTidLayers[TIDDetId(id).wheel()-1];
    }else if( TIDDetId(id).side() == 2 ) {
      return thePosTidLayers[TIDDetId(id).wheel()-1];
    }
    break;

  case StripSubdetector::TEC:
    //edm::LogInfo(TkDetLayers) << "TEC wheel n: " << TECDetId(id).wheel() ;
    if(TECDetId(id).side() ==1 ) {
      return theNegTecLayers[TECDetId(id).wheel()-1];
    }else if( TECDetId(id).side() == 2 ) {
      return thePosTecLayers[TECDetId(id).wheel()-1];
    }
    break;

  case PixelSubdetector::PixelBarrel:
    //edm::LogInfo(TkDetLayers) << "PixelBarrel layer n: " << PXBDetId(id).layer() ;
    return thePixelBarrelLayers[PXBDetId(id).layer()-1];
    break;

  case PixelSubdetector::PixelEndcap:
    //edm::LogInfo(TkDetLayers) << "PixelEndcap disk n: " << PXFDetId(id).disk() ;
    if(PXFDetId(id).side() ==1 ) {
      return theNegPixelForwardLayers[PXFDetId(id).disk()-1];
    }else if( PXFDetId(id).side() == 2 ) {
      return thePosPixelForwardLayers[PXFDetId(id).disk()-1];
    }
    break;

  default:    
    edm::LogError("TkDetLayers") << "ERROR:layer not found!" ;
    // throw(something);
  }
  return 0; //just to avoid compile warnings
}
