#ifndef FrameChanger_H
#define FrameChanger_H

#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"

class FrameChanger {
public:

    typedef ReferenceCountingPointer<Plane>     PlanePtr;

    template <typename T>
    PlanePtr transformPlane( const Plane& plane, const GloballyPositioned<T>& frame) {
	Surface::Base newFrame = toFrame( plane, frame);
	return PlanePtr( new Plane( newFrame.position(), newFrame.rotation()));
    }


/** Moves the first argument ("plane") to the reference frame given by the second 
 *  argument ("frame"). The returned frame is not positioned globally!
 */
    template <typename T, typename U>
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
