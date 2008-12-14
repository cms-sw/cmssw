
#include <vector>
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"

template <class T> T sqr( T t) {return t*t;}

#include "RecoTracker/TkMSParametrization/interface/MSLayersKeeper.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayersKeeperX0AtEta.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayersKeeperX0Averaged.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayersKeeperX0DetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayersAtAngle.h"

//#include "Utilities/Notification/interface/TimingReport.h"
//#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

using namespace std;

const float MultipleScatteringParametrisation::x0ToSigma = 0.0136;

//using namespace PixelRecoUtilities;
//----------------------------------------------------------------------
MultipleScatteringParametrisation::
MultipleScatteringParametrisation( const DetLayer* layer,const edm::EventSetup &iSetup, X0Source x0Source)
 
{
  switch (x0Source) {
  case useX0AtEta: { 
    static MSLayersKeeperX0AtEta x0AtEta; 
    theLayerKeeper = &x0AtEta;
    break;
  }
  case useX0DataAveraged: {
    static MSLayersKeeperX0Averaged x0Averaged;
    theLayerKeeper = &x0Averaged;
    break;
  }

  case useDetLayer: {
    static MSLayersKeeperX0DetLayer x0DetLayer;
    theLayerKeeper = &x0DetLayer;
    break;
  }
  default:
    cout << "** MultipleScatteringParametrisation ** wrong x0Source"<<endl;
    return;
  }
  theLayerKeeper->init(iSetup);
  if (!layer) return;
  theLayer = theLayerKeeper->layer(layer);
} 

//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(
    float pT, float cotTheta) const
{
//   static TimingReport::Item * theTimer =
//      initTiming("MultScattering from vertex",5);
//   TimeMe tm( *theTimer, false);
  float sumX0D = theLayer.sumX0D(cotTheta); 
  return x0ToSigma * sumX0D /pT;
}

//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(
  float pT, float cotTheta, const PixelRecoPointRZ & pointI) const
{
//   static TimingReport::Item * theTimer =
//       initTiming("MultScattering 1p constraint",5);
//   TimeMe tm( *theTimer, false);

  PixelRecoLineRZ lineIO(pointI,cotTheta);
  PixelRecoPointRZ pointO = theLayer.crossing(lineIO).first;

  const MSLayersAtAngle & layersAtEta = theLayerKeeper->layers(cotTheta);
  
  float sumX0D = layersAtEta.sumX0D(pointI, pointO);
  return x0ToSigma * sumX0D /pT;
}

//----------------------------------------------------------------------
float MultipleScatteringParametrisation::operator()(
    float pT,
    const PixelRecoPointRZ & pointI,
    const PixelRecoPointRZ & pointO,
    Consecutive consecutive) const
{   
//   static TimingReport::Item * theTimer =
//       initTiming("MultScattering 2p constraint",5);
//   TimeMe tm( *theTimer, false);


  PixelRecoLineRZ lineIO(pointI, pointO);
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
