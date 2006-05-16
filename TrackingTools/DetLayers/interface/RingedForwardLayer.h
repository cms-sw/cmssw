#ifndef DetLayer_RingedForwardLayer_H
#define DetLayer_RingedForwardLayer_H

/** \class RingedForwardLayer
 *  Abstract class for a disk composed of rings of detectors.
 */

#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

class ForwardDetRing;
class ForwardDetRingBuilder;

class RingedForwardLayer : public ForwardDetLayer {

public:
  
  RingedForwardLayer() {};

//   RingedForwardLayer( std::vector<const GeomDet*>::const_iterator first,
// 		         std::vector<const GeomDet*>::const_iterator last);

//   RingedForwardLayer( const std::vector<const GeomDet*>& dets);


  virtual ~RingedForwardLayer();


  //--- GeometricSearchDet interface
//   virtual std::vector<const GeometricSearchDet*> components() const {return theDets;}

  //--- Extension of the interface
//   virtual GeomDet* operator()( double x, double phi) const =0;

//   virtual void addDets( detunit_p_iter ifirst, detunit_p_iter ilast) {}
  
private:  
//   std::vector<const GeomDet*> theDets;

};
#endif

