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

  RingedForwardLayer( vector<const Det*>::const_iterator first,
		      vector<const Det*>::const_iterator last);

  RingedForwardLayer( const vector<const Det*>& dets);


  virtual ~RingedForwardLayer();


  //--- GeometricSearchDet interface
  virtual vector<const GeometricSearchDet*> components() const {return theDets;}

  //--- Extension of the interface
  /*
  virtual Module module() const;

  virtual Det* operator()( double x, double phi) const {return 0;}

  virtual void addDets( detunit_p_iter ifirst, detunit_p_iter ilast) {}
  */
  
private:  
  vector<const Det*> theDets;

};
#endif

