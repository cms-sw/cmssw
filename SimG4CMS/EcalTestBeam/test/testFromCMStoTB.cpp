#include "DataFormats/Math/interface/Point3D.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
 

#include <vector>
#include <cmath>

using namespace std;

void computeRotation(double & myTheta, double & myPhi, ROOT::Math::Rotation3D & CMStoTB, ROOT::Math::Rotation3D & TBtoCMS) {

  // rotation matrix to move from the CMS reference frame to the test beam one
  
  ROOT::Math::Rotation3D * fromCMStoTB = new ROOT::Math::Rotation3D();

  // rotation matrix to move from the test beam reference frame to the CMS one

  ROOT::Math::Rotation3D * fromTBtoCMS = new ROOT::Math::Rotation3D();

  double angle1 = 90.*deg - myPhi;
  ROOT::Math::RotationZ * r1 = new ROOT::Math::RotationZ(angle1);
  double angle2 = myTheta;
  ROOT::Math::RotationX * r2 = new ROOT::Math::RotationX(angle2);
  double angle3 = 90.*deg;
  ROOT::Math::RotationZ * r3 = new ROOT::Math::RotationZ(angle3);
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
  ROOT::Math::RotationZ * r11 = new ROOT::Math::RotationZ(angle11);
  double angle12 = -myTheta;
  ROOT::Math::RotationX * r12 = new ROOT::Math::RotationX(angle12);
  double angle13 = -90.*deg;
  ROOT::Math::RotationZ * r13 = new ROOT::Math::RotationZ(angle13);

  (*fromTBtoCMS) *= (*r11);
  (*fromTBtoCMS) *= (*r12);
  (*fromTBtoCMS) *= (*r13);
  
  cout 
    << "Rotation matrix from test beam to CMS frame = " << (*fromTBtoCMS)
    << "built from: \n" 
    << " the rotation of " << angle13 << " around Z " << (*r13) << "\n"
    << " the rotation of " << angle12 << " around X " << (*r12) << "\n"
    << " the rotation of " << angle11 << " around Z " << (*r11) << std::endl;

  ROOT::Math::Rotation3D test = (*fromCMStoTB)*(*fromTBtoCMS);
  
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
  
  double xx = -cos(myTheta)*cos(myPhi);
  double xy = -cos(myTheta)*sin(myPhi);
  double xz = sin(myTheta);
  
  double yx = sin(myPhi);
  double yy = -cos(myPhi);
  double yz = 0.;
  
  double zx = sin(myTheta)*cos(myPhi);
  double zy = sin(myTheta)*sin(myPhi);
  double zz = cos(myTheta);

  ROOT::Math::Rotation3D * fromCMStoTB = new ROOT::Math::Rotation3D(xx, xy, xz, yx, yy, yz, zx, zy, zz);

  cout << "Total rotation matrix from CMS to test beam frame = " << (*fromCMStoTB) << endl;

  // rotation matrix to move from the test beam reference frame to the CMS one

  xx = -cos(myTheta)*cos(myPhi);
  xy = sin(myPhi);
  xz = sin(myTheta)*cos(myPhi);

  yx = -cos(myTheta)*sin(myPhi);
  yy = -cos(myPhi);
  yz = sin(myTheta)*sin(myPhi);
  
  zx = sin(myTheta);
  zy = 0.;
  zz = cos(myTheta);

  ROOT::Math::Rotation3D * fromTBtoCMS = new ROOT::Math::Rotation3D(xx, xy, xz, yx, yy, yz, zx, zy, zz);

  cout << "Total rotation matrix from test beam to CMS frame = " << (*fromTBtoCMS) << endl;

  ROOT::Math::Rotation3D test = (*fromCMStoTB)*(*fromTBtoCMS);
  
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
  
  ROOT::Math::Rotation3D * CMStoTB = new ROOT::Math::Rotation3D();
  ROOT::Math::Rotation3D * TBtoCMS = new ROOT::Math::Rotation3D();

  checkTotalRotation(myTheta, myPhi);

  computeRotation(myTheta, myPhi, (*CMStoTB), (*TBtoCMS) );
  cout << "\n===========================================\n" << endl;

  double xx = 0.; double yy = 0. ; double zz = 0.;
  math::XYZPoint test(xx, yy, zz);
  double newTheta = myTheta;
  double newPhi = myPhi;


  for ( int ieta = -1; ieta <= 1; ++ieta ) {
    for ( int iphi = -1; iphi <= 1; ++iphi ) {
      
      newTheta = myTheta+(double)ieta;
      newPhi = myPhi+(double)iphi;

      xx = myMod*sin(newTheta)*cos(newPhi);
      yy = myMod*sin(newTheta)*sin(newPhi);
      zz = myMod*cos(newTheta);
      test.SetCoordinates(xx,yy,zz);
  
      cout << "\n ieta = " << ieta << " iphi = " << iphi << endl;
      cout << "\n From CMS to TB \n" << endl;
      cout << "Input vector  = " << test 
           << " corresponding to theta = " 
           << newTheta << " phi = " << newPhi << endl;
      
      math::XYZPoint testrot = (*CMStoTB)*test;
  
      cout << "Output vector = " << testrot 
           << " corresponding to theta = " 
           << testrot.theta() << " phi = " << testrot.phi() << endl; 
      
      cout << "\n From TB to CMS \n" << endl;
      
      math::XYZPoint thistest = (*TBtoCMS)*testrot;
      
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
      test.SetCoordinates(xx,yy,zz);

      cout << "\n ix = " << ix << " iy = " << iy << endl;
      cout << "\n From TB to CMS \n" << endl;
      cout << "Input vector  = " << test << endl;
      
      math::XYZPoint testrot = (*TBtoCMS)*test;
  
      cout << "Output vector = " << testrot 
           << " corresponding to theta = " 
           << testrot.theta() << " phi = " << testrot.phi() << endl; 
      
      cout << "\n From CMS to TB \n" << endl;
      
      math::XYZPoint thistest = (*CMStoTB)*testrot;
      
      cout << "Output vector = " << thistest << endl;
      cout << "\n===========================================\n" << endl;
  
    }
  }

  return 0;

}
