#ifndef TkNavigation_DiskLessInnerRadius_H
#define TkNavigation_DiskLessInnerRadius_H

#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "DataFormats/GeometrySurface/interface/BoundDisk.h"
#include <functional>

/** less predicate for disks based on the inner radius
 */

class DiskLessInnerRadius : 
  public std::binary_function<const ForwardDetLayer*,const ForwardDetLayer*,bool>
{
public:
  bool operator()( const ForwardDetLayer* a, const ForwardDetLayer* b) {
    return a->specificSurface().innerRadius() < 
           b->specificSurface().innerRadius();
  }
};

#endif // DiskLessInnerRadius_H
