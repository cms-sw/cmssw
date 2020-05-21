#ifndef TkDetLayers_PixelBladeBuilder_h
#define TkDetLayers_PixelBladeBuilder_h

//#include "PixelBlade.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/GeometrySurface/interface/TkRotation.h"

/** A concrete builder for PixelBlade
 */

#pragma GCC visibility push(hidden)
template <class T>
class PixelBladeBuilder {
public:
  PixelBladeBuilder(){};

  T* build(const GeometricDet* geometricDetFrontPanel,
           const GeometricDet* geometricDetBackPanel,
           const TrackerGeometry* theGeomDetGeometry) __attribute__((cold));
};

template <class T>
T* PixelBladeBuilder<T>::build(const GeometricDet* geometricDetFrontPanel,
                               const GeometricDet* geometricDetBackPanel,
                               const TrackerGeometry* theGeomDetGeometry)

{
  std::vector<const GeometricDet*> frontGeometricDets = geometricDetFrontPanel->components();
  std::vector<const GeometricDet*> backGeometricDets = geometricDetBackPanel->components();

  std::vector<const GeomDet*> theFrontGeomDets;
  std::vector<const GeomDet*> theBackGeomDets;

  for (auto& frontGeometricDet : frontGeometricDets) {
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(frontGeometricDet->geographicalID());
    theFrontGeomDets.push_back(theGeomDet);
  }

  for (auto& backGeometricDet : backGeometricDets) {
    const GeomDet* theGeomDet = theGeomDetGeometry->idToDet(backGeometricDet->geographicalID());
    theBackGeomDets.push_back(theGeomDet);
  }

  //edm::LogInfo(TkDetLayers) << "FrontGeomDet.size(): " << theFrontGeomDets.size() ;
  //edm::LogInfo(TkDetLayers) << "BackGeomDet.size():  " << theBackGeomDets.size() ;

  return new T(theFrontGeomDets, theBackGeomDets);
}

#pragma GCC visibility pop
#endif
