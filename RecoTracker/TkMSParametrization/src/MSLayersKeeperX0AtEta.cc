#include "MSLayersKeeperX0AtEta.h"
#include "MultipleScatteringX0Data.h"
#include "MultipleScatteringGeometry.h"
#include <algorithm>
#include "DataFormats/Math/interface/approx_log.h"

using namespace std;

namespace {
  template <class T> inline T sqr( T t) {return t*t;}
  // float unsafe_asinhf(float x) { return std::abs(x)>10.f ? std::copysign(3.f,x) : unsafe_logf<3>(x+std::sqrt(1.f+x*x)); }
  inline float unsafe_asinhf(float x) { return unsafe_logf<3>(x+std::sqrt(1.f+x*x)); }
}

//------------------------------------------------------------------------------
const MSLayersAtAngle & MSLayersKeeperX0AtEta::layers(float cotTheta) const
{
  float eta = unsafe_asinhf(cotTheta);
  return theLayersData[idxBin(eta)];
}

//------------------------------------------------------------------------------
float MSLayersKeeperX0AtEta::eta(int idxBin) const
{ return (idxBin+0.5-theHalfNBins)*theDeltaEta; }

//------------------------------------------------------------------------------
int MSLayersKeeperX0AtEta::idxBin(float eta) const
{
  float ieta = eta/theDeltaEta;
  if ( std::abs(ieta) >= theHalfNBins - 1.e-3)
    return (eta>0) ? max(2*theHalfNBins-1,0) : 0;
  else
    return int(ieta+theHalfNBins);
}

//------------------------------------------------------------------------------
void MSLayersKeeperX0AtEta::init(const edm::EventSetup &iSetup)
{
  if (isInitialised) return;
  isInitialised = true;
  const float BIG = 99999.;

  // set size from data file
  MultipleScatteringX0Data msX0data;
  theHalfNBins = msX0data.nBinsEta();
  float etaMax = msX0data.maxEta();
  theDeltaEta = (theHalfNBins!=0) ? etaMax/theHalfNBins : BIG;

  theLayersData = vector<MSLayersAtAngle>(max(2*theHalfNBins, 1));
  MultipleScatteringGeometry layout(iSetup);
  for (int idxbin = 0; idxbin < 2*theHalfNBins; idxbin++) {
    float etaValue = eta(idxbin);
    float cotTheta = sinh(etaValue);

    vector<MSLayer> layers = layout.detLayers(etaValue,0,iSetup);
    vector<MSLayer> tmplay = layout.otherLayers(etaValue,iSetup);
    layers.insert(layers.end(),tmplay.begin(),tmplay.end()); 
    sort(layers.begin(), layers.end()); 
    setX0(layers, etaValue, msX0data);
    theLayersData[idxbin] = MSLayersAtAngle(layers);
    PixelRecoPointRZ zero(0.,0.);
    PixelRecoLineRZ line( zero, cotTheta);
    vector<MSLayer>::iterator it;
    for (it = layers.begin(); it != layers.end(); it++) {
      float x0 = getDataX0(*it).x0;
      float sumX0D = theLayersData[idxbin].sumX0D(zero, 
          it->crossing(line).first); 
      setDataX0(*it, DataX0(x0, sumX0D, cotTheta));
      theLayersData[idxbin].update(*it);
    } 
  }

  // add layers not seen from nominal vertex but crossed if
  // vertex seperated from nominal by less than 3 sigma
  for (int idxbin = 0; idxbin < 2*theHalfNBins; idxbin++) {
    float etaValue = eta(idxbin);
    for (int isign=-1; isign <=1; isign+=2) {
      float z = isign*15.9;   //3 sigma from zero
      const MSLayersAtAngle & layersAtAngle = theLayersData[idxbin];
      vector<MSLayer> candidates = layout.detLayers( etaValue, z,iSetup);
      vector<MSLayer>::iterator it;
      for (it = candidates.begin(); it != candidates.end(); it++) {
        if (layersAtAngle.findLayer(*it)) continue;
        const MSLayer * found = 0;
        int bin = idxbin;
        while(!found) {
          bin--; if (bin < 0) break; 
          found = theLayersData[bin].findLayer(*it);
        }
        bin = idxbin;
        while(!found) {
          bin++; if (bin > 2*theHalfNBins-1) break;
          found = theLayersData[bin].findLayer(*it);
        }
        if (found) theLayersData[idxbin].update(*found);
      }
    }
  }

// cout << "LAYERS, size=: "<<theLayersData.size()<< endl;
/*
  for (int idxbin = 0; idxbin <= theHalfNBins; idxbin+=25) {
    float etaValue = eta(idxbin);
    const MSLayersAtAngle & layers= theLayersData[idxbin];
    cout << "ETA: "<< etaValue <<" (bin:"<<idxbin<<") #layers:"
         <<layers.size()<<endl;
    layers.print();
  }
  for (int idxbin = 2*theHalfNBins-1; idxbin > theHalfNBins; idxbin-=25) {
    float etaValue = eta(idxbin);
    const MSLayersAtAngle & layers= theLayersData[idxbin];
    cout << "ETA: "<< etaValue <<" (bin:"<<idxbin<<") #layers:"
         <<layers.size()<<endl;
    layers.print();
  }
*/
}

//------------------------------------------------------------------------------
void MSLayersKeeperX0AtEta::setX0(
    vector<MSLayer>& layers, 
    float eta, 
    const SumX0AtEtaDataProvider & sumX0)
{
  const float BIG = 99999.;
  float cotTheta = sinh(eta);
  float sinTheta = 1/sqrt(1+sqr(cotTheta));
  float cosTheta = cotTheta*sinTheta;

  float sumX0atAngleLast = 0.;
  vector<MSLayer>::iterator il;
  for (il = layers.begin(); il != layers.end(); il++) {
    PixelRecoLineRZ line(PixelRecoPointRZ(0.,0.), cotTheta);
    float rL= (*il).crossing(line).first.r();
    float rN = (il+1 != layers.end()) ? (il+1)->crossing(line).first.r() : BIG;
    float rBound = (rL+rN)/2.;
    float sumX0atAngle = sumX0.sumX0atEta(eta,rBound);
   
    float dX0 = (il->face() == GeomDetEnumerators::barrel) ?
      (sumX0atAngle - sumX0atAngleLast)*sinTheta
      : (sumX0atAngle - sumX0atAngleLast)* fabs(cosTheta);
   
  setDataX0(*il,DataX0(dX0,0,cotTheta) );
    sumX0atAngleLast = sumX0atAngle;
  }
}
