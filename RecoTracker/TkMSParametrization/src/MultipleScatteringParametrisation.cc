
#include <vector>
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"


template <class T> inline T sqr( T t) {return t*t;}

#include "MSLayersKeeper.h"
#include "MSLayersKeeperX0AtEta.h"
#include "MSLayersKeeperX0Averaged.h"
#include "MSLayersKeeperX0DetLayer.h"
#include "MSLayersAtAngle.h"

//#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

#include<iostream>

using namespace std;

const float MultipleScatteringParametrisation::x0ToSigma = 0.0136f;

namespace {
  struct Keepers {
    MSLayersKeeperX0DetLayer x0DetLayer;
    MSLayersKeeperX0AtEta x0AtEta;
    MSLayersKeeperX0Averaged x0Averaged;
    MSLayersKeeper * keepers[3];// {&x0DetLayer,&x0AtEta,&x0Averaged};
    bool isInitialised; // =false;
    MSLayersKeeper const * operator()(int i) const { return keepers[i];}
    void init(const edm::EventSetup &iSetup) {
      if (isInitialised) return;
      for (auto x : keepers) x->init(iSetup);
      isInitialised=true;
    }
    Keepers() : keepers{&x0DetLayer,&x0AtEta,&x0Averaged}, isInitialised(false) {}
  };

  const Keepers keepers;

}

void MultipleScatteringParametrisation::initKeepers(const edm::EventSetup &iSetup){
  const_cast<Keepers&>(keepers).init(iSetup);
}

//using namespace PixelRecoUtilities;
//----------------------------------------------------------------------
MultipleScatteringParametrisation::
MultipleScatteringParametrisation( const DetLayer* layer,const edm::EventSetup &iSetup, X0Source x0Source) :
  theLayerKeeper(keepers(x0Source))
{

  // FIXME not thread safe: move elsewhere...
  initKeepers(iSetup);

  if (!layer) return;
  theLayer = theLayerKeeper->layer(layer);
} 

//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(
    float pT, float cotTheta, float) const
{
  float sumX0D = theLayer.sumX0D(cotTheta); 
  return x0ToSigma * sumX0D /pT;
}

//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(
  float pT, float cotTheta, const PixelRecoPointRZ & pointI, float tip) const
{

  PixelRecoLineRZ lineIO(pointI, cotTheta, tip);
  PixelRecoPointRZ pointO = theLayer.crossing(lineIO).first;

  const MSLayersAtAngle & layersAtEta = theLayerKeeper->layers(cotTheta);
  
  float sumX0D = layersAtEta.sumX0D(pointI, pointO);
  return x0ToSigma * sumX0D /pT;
}


float 
MultipleScatteringParametrisation::operator()(float pT, float cotTheta, const PixelRecoPointRZ & pointI,  int il) const {

  PixelRecoLineRZ lineIO(pointI, cotTheta);
  PixelRecoPointRZ pointO = theLayer.crossing(lineIO).first;

  const MSLayersAtAngle & layersAtEta = theLayerKeeper->layers(cotTheta);
  
  float sumX0D = layersAtEta.sumX0D(il, theLayer.seqNum(), pointI, pointO);
  return x0ToSigma * sumX0D /pT;
}


//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(
    float pT,
    const PixelRecoPointRZ & pointI,
    const PixelRecoPointRZ & pointO,
    Consecutive consecutive,
    float tip) const
{   


  PixelRecoLineRZ lineIO(pointI, pointO, tip);
  PixelRecoPointRZ pointM = theLayer.crossing(lineIO).first;
  float cotTheta = lineIO.cotLine();

  if (consecutive==useConsecutive) {
    float dist = fabs(  (pointO.r()-pointM.r())
                      * (pointM.r()-pointI.r())
                      / (pointO.r()-pointI.r()) );
    return  x0ToSigma * sqrt(theLayer.x0(cotTheta)) * dist /pT;
  } else {
    const MSLayersAtAngle & layersAtEta = theLayerKeeper->layers(cotTheta);
    float sumX0D = layersAtEta.sumX0D(pointI, pointM, pointO);
    return x0ToSigma * sumX0D /pT;
  }
}

float MultipleScatteringParametrisation::operator()(
    float pT,
    const PixelRecoPointRZ & pointV,
    const PixelRecoPointRZ & pointO,
    int ol) const
{   

  PixelRecoLineRZ lineIO(pointV, pointO);
  PixelRecoPointRZ pointI = theLayer.crossing(lineIO).first;
  float cotTheta = lineIO.cotLine();

  const MSLayersAtAngle & layersAtEta = theLayerKeeper->layers(cotTheta);
  float sumX0D = layersAtEta.sumX0D(pointV.z(), theLayer.seqNum(), ol, pointI, pointO);
  return x0ToSigma * sumX0D /pT;
}
