#include "TrackPropagation/RungeKutta/interface/PathToPlane2Order.h"
#include "TrackPropagation/RungeKutta/interface/RKLocalFieldProvider.h"
#include "TrackPropagation/RungeKutta/interface/FrameChanger.h"
#include "TrackingTools/GeomPropagators/interface/HelixArbitraryPlaneCrossing.h"

#include <iostream>

std::pair<bool,double> 
PathToPlane2Order::operator()( const Plane& plane, 
			       const Vector3D& pos,
			       const Vector3D& momentum,
			       double charge,
			       const PropagationDirection propDir)
{
  // access to the field in field frame local coordinates
    RKLocalFieldProvider::Vector B = theField.inTesla( pos.x(), pos.y(), pos.z());

    // Frame::GlobalVector localZ = Frame::GlobalVector( B.unit()); // local Z along field
    // transform field axis to global frame
    Frame::GlobalVector localZ = theFieldFrame->toGlobal( Frame::LocalVector( B.unit())); // local Z along field

    Frame::GlobalVector localY = localZ.cross( Frame::GlobalVector( 1,0,0));
    if (localY.mag() < 0.1) {
	localY = localZ.cross( Frame::GlobalVector(0,1,0)).unit();
    }
    else {
	localY = localY.unit();
    }
    Frame::GlobalVector localX = localY.cross(localZ);


    Frame::PositionType fpos( theFieldFrame->toGlobal( Frame::LocalPoint(pos)));
    Frame::RotationType frot( localX, localY, localZ);
    // frame in which the field is along Z
    Frame frame( fpos, frot);
    
    //    cout << "PathToPlane2Order frame " << frame.position() << endl << frame.rotation() << endl;
    
    // transform the position and direction to that frame
    Frame::LocalPoint localPos = frame.toLocal( fpos); // same as LocalPoint(0,0,0)

    //transform momentum from field frame to new frame via global frame
    Frame::GlobalVector gmom( theFieldFrame->toGlobal( Frame::LocalVector(momentum)));
    Frame::LocalVector localMom = frame.toLocal( gmom); 

    // transform the plane to the same frame
    FrameChanger::Plane localPlane =  FrameChanger::transformPlane( plane, frame);
/*
     cout << "PathToPlane2Order input plane       " << plane.position() << endl 
 	 << plane.rotation() << endl;
     cout << "PathToPlane2Order transformed plane " << localPlane->position() << endl 
 	 << localPlane->rotation() << endl;
*/
    double k = 2.99792458e-3;
    double transverseMomentum = localMom.perp();   // transverse to the field
    if (!(transverseMomentum != 0) ) {  // if (!(x!=0)) will trap both 0 and NaN
      //LogDebug("PathToPlane2Order_ZeroMomentum") << "Momentum transverse to the field is zero or Nan (" << transverseMomentum << ")\n";
        return std::pair<bool,double>(false,0);
    }
    double curvature = -k * charge * B.mag() / transverseMomentum;
/*
     cout << "PathToPlane2Order curvature " << curvature << endl;
     cout << "transverseMomentum " << transverseMomentum << endl;
     cout << "B.mag() " << B.mag() << endl;
     cout << "localZ " << localZ << endl;
     cout << "pos      " << pos << endl;
     cout << "momentum " << momentum << endl;
     cout << "localPos " << localPos << endl;
     cout << "localMom " << localMom << endl;
*/
/*
    cout << "PathToPlane2Order: local pos " << localPos << " mom " << localMom 
	 << " curvature " << curvature << endl;
    cout << "PathToPlane2Order: local plane pos " << localPlane->position() 
	 << " normal " << localPlane->normalVector() << endl;
*/
    HelixArbitraryPlaneCrossing crossing( localPos.basicVector(), localMom.basicVector(), 
					  curvature, propDir);
    std::pair<bool,double> res = crossing.pathLength(localPlane);

    return res;
}
