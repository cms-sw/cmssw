#include <cmath>
#include "SimG4Core/CustomPhysics/interface/Decay3Body.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/EulerAngles.h"
#include "Math/GenVector/Boost.h"
#include <vector> 
#include <cmath>

#include "Randomize.hh"

Decay3Body::Decay3Body() {
}

Decay3Body::~Decay3Body() {
}


void Decay3Body::doDecay(const G4LorentzVector & mother,
                               G4LorentzVector & daughter1,
                               G4LorentzVector & daughter2,
                               G4LorentzVector & daughter3) {

  double m0 = mother.m();
  double m1 = daughter1.m();
  double m2 = daughter2.m();
  double m3 = daughter3.m();
  double sumM2 = m0*m0 + m1*m1 + m2*m2 + m3*m3;
  double tolerance = 1.0e-9;

  math::XYZTLorentzVectorD mmm(mother.px(),mother.py(),mother.pz(),mother.e());

  if (m0 < m1+m2+m3) {
	std::cout << "Error: Daughters too heavy!" << std::endl;
    std::cout << "M: " << m0/GeV <<
       " < m1+m2+m3: " << m1/GeV + m2/GeV + m3/GeV << std::endl;
    return;
  } else {
    double m2_12max = sqr(m0-m3);
    double m2_12min = sqr(m1+m2);
    double m2_23max = sqr(m0-m1);
    double m2_23min = sqr(m2+m3);

    double x1,x2;
    double m2_12 = 0.0;
    double m2_23 = 0.0;
    double E2_12,E3_12;
	double m2_23max_12,m2_23min_12;

    do {
// Pick values for m2_12 and m2_23 uniformly:
      x1 = G4UniformRand();
      m2_12 = m2_12min + x1*(m2_12max-m2_12min);
      x2 = G4UniformRand();
      m2_23 = m2_23min + x2*(m2_23max-m2_23min);

// From the allowed range of m2_23 (given m2_12), determine if the point is valid:
// (formulae taken from PDG booklet 2004 kinematics, page 305, Eqs. 38.22a+b)
      E2_12 = (m2_12 - m1*m1 + m2*m2)/(2.0*sqrt(m2_12));
      E3_12 = (m0*m0 - m2_12 - m3*m3)/(2.0*sqrt(m2_12));
      m2_23max_12 = sqr(E2_12+E3_12)-sqr(sqrt(sqr(E2_12)-m2*m2)-sqrt(sqr(E3_12)-m3*m3));
      m2_23min_12 = sqr(E2_12+E3_12)-sqr(sqrt(sqr(E2_12)-m2*m2)+sqrt(sqr(E3_12)-m3*m3));
    } while ((m2_23 > m2_23max_12) || (m2_23 < m2_23min_12));

// Determine the value of the third invariant mass squared:
    double m2_13 = sumM2 - m2_12 - m2_23;

// Calculate the energy and size of the momentum of the three daughters:  
    double e1 = (m0*m0 + m1*m1 - m2_23)/(2.0*m0);
    double e2 = (m0*m0 + m2*m2 - m2_13)/(2.0*m0);
    double e3 = (m0*m0 + m3*m3 - m2_12)/(2.0*m0);
    double p1 = sqrt(e1*e1 - m1*m1);
    double p2 = sqrt(e2*e2 - m2*m2);
    double p3 = sqrt(e3*e3 - m3*m3);

// Calculate cosine of the relative angles between the three daughters:
    double cos12 = (m1*m1 + m2*m2 + 2.0*e1*e2 - m2_12)/(2.0*p1*p2);
    double cos13 = (m1*m1 + m3*m3 + 2.0*e1*e3 - m2_13)/(2.0*p1*p3);
    double cos23 = (m2*m2 + m3*m3 + 2.0*e2*e3 - m2_23)/(2.0*p2*p3);
    if (fabs(cos12) > 1.0) std::cout << "Error: Undefined angle12!" << std::endl;
    if (fabs(cos13) > 1.0) std::cout << "Error: Undefined angle13!" << std::endl;
    if (fabs(cos23) > 1.0) std::cout << "Error: Undefined angle23!" << std::endl;

// Find the four vectors of the particles in a chosen (i.e. simple) frame:
    double xi    = 2.0 * pi * G4UniformRand();
    math::XYZVectorD q1(0.0,0.0,p1);
    math::XYZVectorD q2( sin(acos(cos12))*cos(xi)*p2, sin(acos(cos12))*sin(xi)*p2,cos12*p2);
    math::XYZVectorD q3(-sin(acos(cos13))*cos(xi)*p3,-sin(acos(cos13))*sin(xi)*p3,cos13*p3);

// Rotate all three daughters momentum with the angles theta and phi:
    double theta = acos(2.0 * G4UniformRand() - 1.0);
    double phi   = 2.0 * pi * G4UniformRand();
    double psi   = 2.0 * pi * G4UniformRand();
    
    ROOT::Math::EulerAngles ang(phi,theta,psi);
    ROOT::Math::Rotation3D rot(ang);

    math::XYZVectorD q1rot = rot*q1;
    math::XYZVectorD q2rot = rot*q2;
    math::XYZVectorD q3rot = rot*q3;

    math::XYZTLorentzVectorD daughter1_orig(q1rot.X(),q1rot.Y(),q1rot.Z(),e1);
    math::XYZTLorentzVectorD daughter2_orig(q2rot.X(),q2rot.Y(),q2rot.Z(),e2);
    math::XYZTLorentzVectorD daughter3_orig(q3rot.X(),q3rot.Y(),q3rot.Z(),e3);

    ROOT::Math::Boost cmboost(mmm.BoostToCM());

    // Check of total angle and momentum:
    if (acos(cos12)+acos(cos13)+acos(cos23)-2.0*pi > tolerance)
      std::cout << "Error: Total angle not 2pi! " <<
	acos(cos12)+acos(cos13)+acos(cos23)-2.0*pi << std::endl;
    if (fabs(daughter1_orig.px()+daughter2_orig.px()+daughter3_orig.px())/GeV > tolerance)
      std::cout << "Error: Total 3B Px not conserved! " << 
	(daughter1_orig.px()+daughter2_orig.px()+daughter3_orig.px())/GeV << std::endl;
    if (fabs(daughter1_orig.py()+daughter2_orig.py()+daughter3_orig.py())/GeV > tolerance)
      std::cout << "Error: Total 3B Py not conserved! " << 
	(daughter1_orig.py()+daughter2_orig.py()+daughter3_orig.py())/GeV << std::endl;
    if (fabs(daughter1_orig.pz()+daughter2_orig.pz()+daughter3_orig.pz())/GeV > tolerance)
      std::cout << "Error: Total 3B Pz not conserved! " << 
	(daughter1.pz()+daughter2.pz()+daughter3.pz())/GeV << std::endl;
    
    // Boost the daughters back to the frame of the mother:

    math::XYZTLorentzVectorD temp1(cmboost(daughter1_orig));
    math::XYZTLorentzVectorD temp2(cmboost(daughter2_orig));
    math::XYZTLorentzVectorD temp3(cmboost(daughter3_orig));
    
    daughter1.setPx(temp1.Px());
    daughter1.setPy(temp1.Py());
    daughter1.setPz(temp1.Pz());
    daughter1.setE(temp1.e());

    daughter2.setPx(temp2.Px());
    daughter2.setPy(temp2.Py());
    daughter2.setPz(temp2.Pz());
    daughter2.setE(temp2.e());

    daughter3.setPx(temp3.Px());
    daughter3.setPy(temp3.Py());
    daughter3.setPz(temp3.Pz());
    daughter3.setE(temp3.e());

    return;
  }
}


double Decay3Body::sqr(double a) {
  return a*a;
}

