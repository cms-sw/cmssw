#include "CLHEP/Vector/ThreeVector.h"
#include "CLHEP/Vector/Rotation.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/SystemOfUnits.h"
 

#include <vector>
#include <cmath>

using namespace std;

void computeRotation(double & myTheta, double & myPhi, HepRotation & CMStoTB, HepRotation & TBtoCMS) {

  // rotation matrix to move from the CMS reference frame to the test beam one
  
  HepRotation * fromCMStoTB = new HepRotation();

  // rotation matrix to move from the test beam reference frame to the CMS one

  HepRotation * fromTBtoCMS = new HepRotation();

  double angle1 = 90.*deg - myPhi;
  HepRotationZ * r1 = new HepRotationZ(angle1);
  double angle2 = myTheta;
  HepRotationX * r2 = new HepRotationX(angle2);
  double angle3 = 90.*deg;
  HepRotationZ * r3 = new HepRotationZ(angle3);
  (*fromCMStoTB) *= (*r3);
  (*fromCMStoTB) *= (*r2);
  (*fromCMStoTB) *= (*r1);
  
  cout 
    << "Rotation matrix from CMS to test beam frame = " << (*fromCMStoTB)
    << "built from: \n" 
    << " the rotation of " << angle1 << " around Z " << (*r1) << "\n"
    << " the rotation of " << angle2 << " around X " << (*r2) << "\n"
    << " the rotation of " << angle3 << " around Z " << (*r3) << std::endl;

  double angle11 = - 90.*deg + myPhi;
  HepRotationZ * r11 = new HepRotationZ(angle11);
  double angle12 = -myTheta;
  HepRotationX * r12 = new HepRotationX(angle12);
  double angle13 = -90.*deg;
  HepRotationZ * r13 = new HepRotationZ(angle13);

  (*fromTBtoCMS) *= (*r11);
  (*fromTBtoCMS) *= (*r12);
  (*fromTBtoCMS) *= (*r13);
  
  cout 
    << "Rotation matrix from test beam to CMS frame = " << (*fromTBtoCMS)
    << "built from: \n" 
    << " the rotation of " << angle13 << " around Z " << (*r13) << "\n"
    << " the rotation of " << angle12 << " around X " << (*r12) << "\n"
    << " the rotation of " << angle11 << " around Z " << (*r11) << std::endl;

  HepRotation test = (*fromCMStoTB)*(*fromTBtoCMS);
  
  cout 
    << "Product of the two rotations: " << test << endl;
  
  
  CMStoTB = (*fromCMStoTB);
  TBtoCMS = (*fromTBtoCMS);

  delete fromCMStoTB;
  delete fromTBtoCMS;
  delete r1;
  delete r2;
  delete r3;

}

void checkTotalRotation(double & myTheta, double & myPhi) {

  // rotation matrix to move from the CMS reference frame to the test beam one
  
  HepRotation * fromCMStoTB = new HepRotation();

  // rotation matrix to move from the test beam reference frame to the CMS one

  HepRotation * fromTBtoCMS = new HepRotation();
  
  double xx = -cos(myTheta)*cos(myPhi);
  double xy = -cos(myTheta)*sin(myPhi);
  double xz = sin(myTheta);
  
  double yx = sin(myPhi);
  double yy = -cos(myPhi);
  double yz = 0.;
  
  double zx = sin(myTheta)*cos(myPhi);
  double zy = sin(myTheta)*sin(myPhi);
  double zz = cos(myTheta);

  const HepRep3x3 mCMStoTB(xx, xy, xz, yx, yy, yz, zx, zy, zz);

  fromCMStoTB->set(mCMStoTB);
  
  cout << "Total rotation matrix from CMS to test beam frame = " << (*fromCMStoTB) << endl;
  
  xx = -cos(myTheta)*cos(myPhi);
  xy = sin(myPhi);
  xz = sin(myTheta)*cos(myPhi);

  yx = -cos(myTheta)*sin(myPhi);
  yy = -cos(myPhi);
  yz = sin(myTheta)*sin(myPhi);
  
  zx = sin(myTheta);
  zy = 0.;
  zz = cos(myTheta);

  const HepRep3x3 mTBtoCMS(xx, xy, xz, yx, yy, yz, zx, zy, zz);

  fromTBtoCMS->set(mTBtoCMS);
  
  cout << "Total rotation matrix from test beam to CMS frame = " << (*fromTBtoCMS) << endl;

  HepRotation test = (*fromCMStoTB)*(*fromTBtoCMS);
  
  cout 
    << "Product of the two rotations: " << test << endl;

}

int main() {

  // (eta,phi) for a crystal

  double myMod = 1.;
  double myEta = 0.971226;
  double myTheta = 2.0*atan(exp(-myEta));
  double myPhi = 0.115052;
  
  cout << "\n===========================================\n" << endl;
  cout << "Input theta = " << myTheta << " phi = " << myPhi << endl;
  cout << "\n===========================================\n" << endl;
  
  HepRotation * CMStoTB = new HepRotation();
  HepRotation * TBtoCMS = new HepRotation();

  checkTotalRotation(myTheta, myPhi);

  computeRotation(myTheta, myPhi, (*CMStoTB), (*TBtoCMS) );
  cout << "\n===========================================\n" << endl;

  double xx = 0.; double yy = 0. ; double zz = 0.;
  Hep3Vector test(xx, yy, zz);
  double newTheta = myTheta;
  double newPhi = myPhi;


  for ( int ieta = -1; ieta <= 1; ++ieta ) {
    for ( int iphi = -1; iphi <= 1; ++iphi ) {
      
      newTheta = myTheta+(double)ieta;
      newPhi = myPhi+(double)iphi;

      xx = myMod*sin(newTheta)*cos(newPhi);
      yy = myMod*sin(newTheta)*sin(newPhi);
      zz = myMod*cos(newTheta);
      test.set(xx,yy,zz);
  
      cout << "\n ieta = " << ieta << " iphi = " << iphi << endl;
      cout << "\n From CMS to TB \n" << endl;
      cout << "Input vector  = " << test 
           << " corresponding to theta = " 
           << newTheta << " phi = " << newPhi << endl;
      
      Hep3Vector testrot = (*CMStoTB)*test;
  
      cout << "Output vector = " << testrot 
           << " corresponding to theta = " 
           << testrot.theta() << " phi = " << testrot.phi() << endl; 
      
      cout << "\n From TB to CMS \n" << endl;
      
      Hep3Vector thistest = (*TBtoCMS)*testrot;
      
      cout << "Output vector = " << thistest 
           << " corresponding to theta = " 
           << thistest.theta() << " phi = " << thistest.phi() << endl; 
      cout << "\n===========================================\n" << endl;
  
    }
  }

  cout << "\n===========================================\n" << endl;

  for ( int ix = -1; ix <= 1; ++ix ) {
    for ( int iy = -1; iy <= 1; ++iy ) {

      xx = (double)ix * 0.01;
      yy = (double)iy * 0.01;
      zz = 1.;
      test.set(xx,yy,zz);

      cout << "\n ix = " << ix << " iy = " << iy << endl;
      cout << "\n From TB to CMS \n" << endl;
      cout << "Input vector  = " << test << endl;
      
      Hep3Vector testrot = (*TBtoCMS)*test;
  
      cout << "Output vector = " << testrot 
           << " corresponding to theta = " 
           << testrot.theta() << " phi = " << testrot.phi() << endl; 
      
      cout << "\n From CMS to TB \n" << endl;
      
      Hep3Vector thistest = (*CMStoTB)*testrot;
      
      cout << "Output vector = " << thistest << endl;
      cout << "\n===========================================\n" << endl;
  
    }
  }

  return 0;

}
