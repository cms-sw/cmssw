#include "SimG4CMS/Tracker/interface/TrackerG4SimHitNumberingScheme.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

TrackerG4SimHitNumberingScheme::TrackerG4SimHitNumberingScheme(const DDCompactView& cpv,
   const GeometricDet& det ){
  ts = new TouchableToHistory(cpv,det);
}
TrackerG4SimHitNumberingScheme::~TrackerG4SimHitNumberingScheme(){
  delete ts;
}
