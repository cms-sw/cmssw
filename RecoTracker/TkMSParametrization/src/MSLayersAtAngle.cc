
#include "MSLayersAtAngle.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"
#include "DataFormats/Math/interface/approx_log.h"

using namespace std;

namespace {
  template <class T> inline T sqr( T t) {return t*t;}
}


void
MSLayersAtAngle::init() {
  std::sort(theLayers.begin(), theLayers.end());
  int i = -1; indeces.clear();
  for ( auto const & l :  theLayers ) {
    ++i;
    int sq = l.seqNum();
    if (sq<0) continue; 
    if (sq>=int(indeces.size())) indeces.resize(sq+1,-1);
    indeces[sq]=i;
  }
}

//------------------------------------------------------------------------------
MSLayersAtAngle::MSLayersAtAngle(const vector<MSLayer> & layers)
  : theLayers(layers)
{ 
  init();
}
//------------------------------------------------------------------------------
const MSLayer * MSLayersAtAngle::findLayer(const MSLayer & layer) const
{
  vector<MSLayer>::const_iterator it =
     find(theLayers.begin(), theLayers.end(), layer);
  return it==theLayers.end() ? 0 : &(*it);  
}

//------------------------------------------------------------------------------
void MSLayersAtAngle::update(const MSLayer & layer)
{
  vector<MSLayer>::iterator it = find(theLayers.begin(),theLayers.end(),layer); 
  if (it == theLayers.end()) {
    theLayers.push_back(layer);
    init();
  } else {
    *it = layer;
  }
}

//------------------------------------------------------------------------------
float MSLayersAtAngle::sumX0D(
    const PixelRecoPointRZ & pointI,
    const PixelRecoPointRZ & pointO) const
{
  LayerItr iO = findLayer(pointO, theLayers.begin(), theLayers.end());
//  cout << "outer Layer: "<<*iO<<endl;
  LayerItr iI = findLayer(pointI, theLayers.begin(), iO);
//  cout << "inner Layer: "<<*iI<<endl;

  return sqrt(sum2RmRn(iI,iO, pointO.r(),
                              SimpleLineRZ(pointI, pointO)));
}


float MSLayersAtAngle::sumX0D( int il, int ol,
			       const PixelRecoPointRZ & pointI,
			       const PixelRecoPointRZ & pointO) const
{
  // if layer not at this angle (WHY???) revert to slow comp
  if  (il>=int(indeces.size()) || ol>=int(indeces.size()) ||
       indeces[il]<0 || indeces[ol] < 0)  return sumX0D(pointI,pointO);

  LayerItr iI = theLayers.begin() + indeces[il];
  LayerItr iO = theLayers.begin() + indeces[ol];


  return sqrt(sum2RmRn(iI,iO, pointO.r(),
                              SimpleLineRZ(pointI, pointO)));
}

//------------------------------------------------------------------------------

bool doPrint=false;

float MSLayersAtAngle::sumX0D(
    const PixelRecoPointRZ & pointI,
    const PixelRecoPointRZ & pointM,
    const PixelRecoPointRZ & pointO) const
{
  LayerItr iO = findLayer(pointO, theLayers.begin(), theLayers.end());
  LayerItr iI = findLayer(pointI, theLayers.begin(), iO);
  LayerItr iM = findLayer(pointM, iI, iO);


  float drOI = pointO.r() - pointI.r();
  float drMO = pointO.r() - pointM.r();
  float drMI = pointM.r() - pointI.r();

  SimpleLineRZ line(pointI, pointO);
  float sum2I = sum2RmRn(iI+1, iM, pointI.r(), line);
  float sum2O = sum2RmRn(iM, iO, pointO.r(), line);

  float sum =  std::sqrt( sum2I* sqr(drMO) + sum2O*sqr(drMI) )/drOI;
 
 // if (iI!=theLayers.begin() )
 // doPrint = ((*iM).seqNum()<0 || (*iO).seqNum()<0); 
  if (doPrint)
  std::cout << "old " << (*iI).seqNum() << " " << iI-theLayers.begin() << ", "
    << (*iM).seqNum() << " " << iM-theLayers.begin() << ", "
    << (*iO).seqNum() << " " << iO-theLayers.begin() << "  "
    << pointI.r() << " " << pointI.z()
     << " " << sum
     << std::endl;
 

  return sum;
}


float MSLayersAtAngle::sumX0D(float zV, int il, int ol, 
			      const PixelRecoPointRZ & pointI,
			      const PixelRecoPointRZ & pointO) const {

  PixelRecoPointRZ pointV(0.f,zV);

  // if layer not at this angle (WHY???) revert to slow comp
  if  (il>=int(indeces.size()) || ol>=int(indeces.size()) ||
       indeces[il]<0 || indeces[ol] < 0)  return sumX0D(pointV,pointI,pointO);

  LayerItr iI = theLayers.begin() + indeces[il];
  LayerItr iO = theLayers.begin() + indeces[ol];



  float drOI = pointO.r();
  float drMO = pointO.r() - pointI.r();
  float drMI = pointI.r();

  SimpleLineRZ line(pointV, pointO);
  float sum2I = sum2RmRn(theLayers.begin()+1, iI, pointV.r(), line);
  float sum2O = sum2RmRn(iI, iO, pointO.r(), line);

  float sum =  std::sqrt( sum2I* sqr(drMO) + sum2O*sqr(drMI) )/drOI;

  if (doPrint)
  std::cout << "new " << il << " " << (*iI).seqNum() << " " << iI-theLayers.begin() << ", "
    << ol << " " << (*iO).seqNum() << " " << iO-theLayers.begin() << "  " << zV
    << " " << sum
    << std::endl;

  return sum;

}


//------------------------------------------------------------------------------
float MSLayersAtAngle::sum2RmRn(
    MSLayersAtAngle::LayerItr i1,
    MSLayersAtAngle::LayerItr i2,
    float rTarget,
    const SimpleLineRZ & line) const
{
  float sum2 = 0.f;
  float cotTh = line.cotLine();
  for (LayerItr it = i1; it < i2; it++) {
    std::pair<PixelRecoPointRZ,bool> cross = it->crossing(line);
    if (cross.second) {
      float x0 = it->x0(cotTh);
      float dr = rTarget-cross.first.r();
      if (x0 > 1.e-5f) dr *= 1.f+0.038f*unsafe_logf<2>(x0); 
      sum2 += x0*dr*dr;
    } 
//  cout << *it << " crossing: "<<cross.second<<endl;
  }
  return sum2;
}
//------------------------------------------------------------------------------
MSLayersAtAngle::LayerItr MSLayersAtAngle::findLayer(
    const PixelRecoPointRZ & point,
    MSLayersAtAngle::LayerItr ibeg,
    MSLayersAtAngle::LayerItr iend) const
{
  const float BIG=99999.f;
  const float EPSILON = 1.e-8f;
  LayerItr theIt = ibeg; float dist = BIG;
  for (LayerItr it = ibeg; it < iend; it++) {
    float d = it->distance2(point);
    if (d < dist) {
      if (d < EPSILON) return it; 
      dist = d;
      theIt = it;
    }
  }
  return theIt;
}

//------------------------------------------------------------------------------
void MSLayersAtAngle::print() const 
{
  for (LayerItr it = theLayers.begin(); it != theLayers.end(); it++) 
    cout <<*it<<endl;
}

