#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Random/RandFlat.h"

#include <vector>
#include <cmath>

using namespace std;

void computeRotation(Hep3Vector & provaCMS) {

  // rotation matrix to move from the CMS reference frame to the test beam one
  const HepRotation * fromCMStoTB;
  
  // find the angle between the vector and z, i.e. theta

  double myTheta = provaCMS.theta();

  // find the axis orthogonal to the plane spanned by z and the vector

  Hep3Vector zAxis(0.,0.,1.);
  Hep3Vector ortho = provaCMS.cross(zAxis);

  fromCMStoTB = new HepRotation(ortho, myTheta);
  
  cout 
    << "Rotation matrix from CMS to test beam frame = " << (*fromCMStoTB)
    << "built from a rotation of angle = " 
    << myTheta << " around the axis " << ortho << endl;
  
  cout << "Reference point                      = " << provaCMS << endl;
  
  Hep3Vector provaTB = (*fromCMStoTB)*provaCMS;
  cout << "Rotated point                        = " << provaTB << endl;

  delete fromCMStoTB;

}

int main() {

  for ( int i = 0 ; i<= 10 ; ++i ) {

    double myMod = 1.;
    double myTheta = RandFlat::shoot(0.,pi);
    double myPhi = RandFlat::shoot(-pi,pi);
    double xx , yy , zz;
    xx = myMod*sin(myTheta)*cos(myPhi);
    yy = myMod*sin(myTheta)*sin(myPhi);
    zz = myMod*cos(myTheta);
    Hep3Vector test(xx, yy, zz);

    cout << "\n===========================================\n" << endl;
    cout << "Input vector = " << test 
         << " corresponding to theta = " 
         << myTheta << " phi = " << myPhi << endl;

    computeRotation(test);
    
  }
  
  return 0;

}
