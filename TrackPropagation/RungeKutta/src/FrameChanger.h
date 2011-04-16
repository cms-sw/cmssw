#ifndef FrameChanger_H
#define FrameChanger_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

class dso_internal FrameChanger {
public:
  /** Moves the first argument ("plane") to the reference frame given by the second 
   *  argument ("frame"). The returned frame is not positioned globally!
   */
    template <typename T>
    static
    Plane transformPlane( const Plane& plane, const GloballyPositioned<T>& frame) {
        typedef GloballyPositioned<T>                  Frame;
	typename Plane::RotationType rot = plane.rotation() * frame.rotation().transposed();
	typename Frame::LocalPoint lpos = frame.toLocal(plane.position());
	typename Plane::PositionType pos( lpos.basicVector()); // cheat!
	return Plane(pos, rot);
    }


/** Moves the first argument ("plane") to the reference frame given by the second 
 *  argument ("frame"). The returned frame is not positioned globally!
 */
    template <typename T, typename U>
    static
    GloballyPositioned<T> toFrame( const GloballyPositioned<T>& plane, 
				   const GloballyPositioned<U>& frame) {
	typedef GloballyPositioned<T>                  Plane;
	typedef GloballyPositioned<U>                  Frame;

	typename Plane::RotationType rot = plane.rotation() * frame.rotation().transposed();
	typename Frame::LocalPoint lpos = frame.toLocal(plane.position());
	typename Plane::PositionType pos( lpos.basicVector()); // cheat!
	return Plane( pos, rot);

    }

};

#endif
