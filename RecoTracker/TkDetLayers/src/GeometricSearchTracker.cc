#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

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

  cout << "------ GeometricSearchTracker constructed with: ------" << endl;
  cout << "n pxlBarLayers: " << this->pixelBarrelLayers().size() << endl;
  cout << "n tibLayers:    " << this->tibLayers().size() << endl;
  cout << "n tobLayers:    " << this->tobLayers().size() << endl;
  cout << "n negPxlFwdLayers: " << this->negPixelForwardLayers().size() << endl;
  cout << "n negPxlFwdLayers: " << this->posPixelForwardLayers().size() << endl;
  cout << "n negTidLayers: " << this->negTidLayers().size() << endl;
  cout << "n posTidLayers: " << this->posTidLayers().size() << endl;
  cout << "n negTecLayers: " << this->negTecLayers().size() << endl;
  cout << "n posTecLayers: " << this->posTecLayers().size() << endl;

  cout << "n barreLayers:  " << this->barrelLayers().size() << endl;
  cout << "n negforwardLayers: " << this->negForwardLayers().size() << endl;
  cout << "n posForwardLayers: " << this->posForwardLayers().size() << endl;


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
GeometricSearchTracker::detLayer( const DetId& id) const
{
  switch(id.subdetId()) {
  case StripSubdetector::TIB:
    //cout << "TIB layer n: " << TIBDetId(id).layer() << endl;
    return theTibLayers[TIBDetId(id).layer()-1];
    break;

  case StripSubdetector::TOB:
    //cout << "TOB layer n: " << TOBDetId(id).layer() << endl;
    return theTobLayers[TOBDetId(id).layer()-1];
    break;

  case StripSubdetector::TID:
    //cout << "TID wheel n: " << TIDDetId(id).wheel() << endl;
    if(TIDDetId(id).side() ==1 ) {
      return theNegTidLayers[TIDDetId(id).wheel()-1];
    }else if( TIDDetId(id).side() == 2 ) {
      return thePosTidLayers[TIDDetId(id).wheel()-1];
    }
    break;

  case StripSubdetector::TEC:
    //cout << "TEC wheel n: " << TECDetId(id).wheel() << endl;
    if(TECDetId(id).side() ==1 ) {
      return theNegTecLayers[TECDetId(id).wheel()-1];
    }else if( TECDetId(id).side() == 2 ) {
      return thePosTecLayers[TECDetId(id).wheel()-1];
    }
    break;

  case PixelSubdetector::PixelBarrel:
    //cout << "PixelBarrel layer n: " << PXBDetId(id).layer() << endl;
    return thePixelBarrelLayers[PXBDetId(id).layer()-1];
    break;

  case PixelSubdetector::PixelEndcap:
    //cout << "PixelEndcap disk n: " << PXFDetId(id).disk() << endl;
    if(PXFDetId(id).side() ==1 ) {
      return theNegPixelForwardLayers[PXFDetId(id).disk()-1];
    }else if( PXFDetId(id).side() == 2 ) {
      return thePosPixelForwardLayers[PXFDetId(id).disk()-1];
    }
    break;

  default:    
    cout << "ERROR:layer not found!" << endl;
    // throw(something);
  }
  return 0; //just to avoid compile warnings
}
