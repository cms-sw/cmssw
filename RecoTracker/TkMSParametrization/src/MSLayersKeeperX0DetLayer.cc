#include "MSLayersKeeperX0DetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundSurface.h"
#include "DataFormats/GeometrySurface/interface/MediumProperties.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include <vector>

using namespace std;

void MSLayersKeeperX0DetLayer::init(const edm::EventSetup &iSetup)
{
  if (isInitialised) return;
  isInitialised = true;
  //  vector<MSLayer> allLayers = MSLayersKeeperX0DetLayerGeom().detLayers();
  //MP
  vector<MSLayer> allLayers;
  theLayersData = MSLayersAtAngle(allLayers);
  
  vector<MSLayer>::iterator it;
  PixelRecoPointRZ zero(0.,0.); 
  for (it = allLayers.begin(); it != allLayers.end(); it++) {
    PixelRecoPointRZ middle = it->face()== GeomDetEnumerators::barrel ?
        PixelRecoPointRZ(it->position(), it->range().mean())
      : PixelRecoPointRZ(it->range().mean(), it->position());

    float cotTheta = PixelRecoLineRZ(zero,middle).cotLine();
    float x0 =  getDataX0(*it).x0;

    DataX0 dataX0;
    if (it->face()== GeomDetEnumerators::barrel) {
      float sumX0D = theLayersData.sumX0D(zero, middle);
      dataX0 = DataX0(x0, sumX0D, cotTheta);      
    } else {
      float hrange= (it->range().max()-it->range().min())/2.;
      float cot1 = it->position()/(it->range().mean()-hrange/2);
      float cot2 = it->position()/(it->range().mean()+hrange/2);
      PixelRecoLineRZ line1(zero,cot1);
      PixelRecoLineRZ line2(zero,cot2);
      float sum1 = theLayersData.sumX0D(zero,it->crossing(line1).first);
      float sum2 = theLayersData.sumX0D(zero,it->crossing(line2).first);
      float slope = (sum2-sum1)/(1/cot2-1/cot1);
      float sumX0D = sum1 + slope*(1/cotTheta-1/cot1);
      dataX0 = DataX0(x0, sumX0D, cotTheta);
      dataX0.setForwardSumX0DSlope(slope);
    }
    setDataX0(*it, dataX0);
    theLayersData.update(*it);
  }
  cout << "MSLayersKeeperX0DetLayer LAYERS: "<<endl;
  theLayersData.print();
}

// vector<MSLayer>
// MSLayersKeeperX0DetLayer::MSLayersKeeperX0DetLayerGeom::detLayers() const
// {

//   vector<MSLayer> result;

//   vector<const DetLayer*>::const_iterator it;
//   for (it = theLayers.begin(); it != theLayers.end(); it++) {

//     //    const DetUnit * du = (*it)->detUnits().front();
//     const GeomDetUnit * du;
//     //MP how access the geomdetunit??
//    const MediumProperties * mp = du->surface().mediumProperties();
//     float x0 = (mp) ? mp->radLen() : 0.03; 
//     cout << "MediumProperties: "<<mp<<" x0="<<x0<<endl;
//     MSLayer layer(*it, DataX0(x0,0,0));
//     result.push_back( layer);
//   }
//   return result;
// } 

