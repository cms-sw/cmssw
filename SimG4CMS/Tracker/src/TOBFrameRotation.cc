#include "SimG4CMS/Tracker/interface/TOBFrameRotation.h"

Local3DPoint TOBFrameRotation::transformPoint(Local3DPoint & point,G4VPhysicalVolume * v=0) const {
    //
    // Strange to say, do not depend anymore on stereoness ;)
    //
    return Local3DPoint(point.x()/cm,point.z()/cm,-point.y()/cm);
}
