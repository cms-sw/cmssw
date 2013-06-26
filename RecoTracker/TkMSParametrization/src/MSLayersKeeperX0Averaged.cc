#include "MSLayersKeeperX0Averaged.h"
#include "MSLayersKeeperX0AtEta.h"
#include "MultipleScatteringGeometry.h"

using namespace std;

void MSLayersKeeperX0Averaged::init(const edm::EventSetup &iSetup)
{
  if (isInitialised) return;
  isInitialised = true;
//  cout << "HERE INITIALISATION! MSLayersKeeperX0Averaged"<<endl;
  MSLayersKeeperX0AtEta layersX0Eta;
  layersX0Eta.init(iSetup);
  MultipleScatteringGeometry geom(iSetup);
  vector<MSLayer> allLayers = geom.detLayers(iSetup);
  vector<MSLayer>::iterator it;
  for (int i=-1;i<=1;i++) {
    float eta = i*(-1.8);
    vector<MSLayer> tmpLayers = geom.otherLayers(eta,iSetup);
    vector<MSLayer>::const_iterator ic;
    for (ic = tmpLayers.begin(); ic != tmpLayers.end(); ic++) {
      it = find(allLayers.begin(), allLayers.end(), *ic);  
      if (it == allLayers.end()) allLayers.push_back(*ic);
    }
  }

  

for (it = allLayers.begin(); it != allLayers.end(); it++) {
    float cotTheta = (it->face()==GeomDetEnumerators::barrel) ?
        it->range().mean()/it->position()
      : it->position()/it->range().mean();

    int nbins = 0;
    float sumX0 = 0.;
    for (int ibin = 0; ibin < 2*layersX0Eta.theHalfNBins; ibin++) { 
      const MSLayersAtAngle & layers = layersX0Eta.theLayersData[ibin];
      const MSLayer * aLayer = layers.findLayer(*it);
      if (aLayer) { nbins++; sumX0 += getDataX0(*aLayer).x0; }
    }
    if ( nbins==0) nbins=1;

    float hrange= (it->range().max()-it->range().min())/2.;
    DataX0 dataX0;
    if (it->face()==GeomDetEnumerators::endcap) {
      float cot1 = it->position()/(it->range().mean()-hrange/2); 
      float cot2 = it->position()/(it->range().mean()+hrange/2); 
      const MSLayer * aLayer1 = layersX0Eta.layers(cot1).findLayer(*it);
      const MSLayer * aLayer2 = layersX0Eta.layers(cot1).findLayer(*it);
      float sum1 = aLayer1 ? aLayer1->sumX0D(cot1) : 0.;
      float sum2 = aLayer2 ? aLayer2->sumX0D(cot2) : 0.;
      float slope = (sum2-sum1)/(1/cot2-1/cot1);
      float sumX0D = sum1 + slope*(1/cotTheta-1/cot1);
      dataX0 = DataX0(sumX0/nbins, sumX0D, cotTheta);
      dataX0.setForwardSumX0DSlope(slope);
    } else {
      float sumX0D = 0;
      int nb=10;
      for (int i=0; i<nb; i++) {
        float cot = (it->range().mean()+(2*i+1-nb)*hrange/nb)/it->position();
        float sin = 1/sqrt(1+cot*cot);
        const MSLayer * aLayer = layersX0Eta.layers(cot).findLayer(*it);
        if (aLayer) sumX0D += aLayer->sumX0D(cot) * sqrt(sin);
      } 
      dataX0 = DataX0(sumX0/nbins, sumX0D/nb, 0);
    }
    setDataX0(*it, dataX0);
  }
  theLayersData = MSLayersAtAngle(allLayers);
//  cout << "MSLayersKeeperX0Averaged - LAYERS:"<<endl;
//  theLayersData.print();
//  cout << "END OF LAYERS"<<endl;
}
