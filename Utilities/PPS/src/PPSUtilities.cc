#include "Utilities/PPS/interface/PPSUtilities.h"
#include "HepMC/GenParticle.h"
#include "H_BeamParticle.h"
#include <CLHEP/Vector/LorentzVector.h>

CLHEP::HepLorentzVector PPSTools::HectorParticle2LorentzVector(H_BeamParticle hp,int direction)
{
     double partP = sqrt(pow(hp.getE(),2)-ProtonMassSQ);
     double theta = sqrt(pow(hp.getTX(),2)+pow(hp.getTY(),2))*urad;
     double pz = partP*cos(theta);
     double px = tan((double)hp.getTX()*urad)*pz;//PartP*sin(theta)*cos(phi);
     double py = tan((double)hp.getTY()*urad)*pz;//partP*sin(theta)*sin(phi);
     pz*=direction;
     return CLHEP::HepLorentzVector(px,py,pz,hp.getE());
}

H_BeamParticle PPSTools::LorentzVector2HectorParticle(CLHEP::HepLorentzVector p)
{
     H_BeamParticle h_p;
     h_p.set4Momentum(p.px(),p.py(),abs(p.pz()),p.e());
     return h_p;
}
void PPSTools::LorentzBoost(HepMC::GenParticle& p, const string& frame)
{
     CLHEP::HepLorentzVector p_out;
     p_out.setPx(p.momentum().px());
     p_out.setPy(p.momentum().py());
     p_out.setPz(p.momentum().pz());
     LorentzBoost(p_out,frame);
     p.set_momentum(HepMC::FourVector(p_out.px(),p_out.py(),p_out.pz(),p_out.e()));
}
void PPSTools::LorentzBoost(H_BeamParticle& h_p, int dir , const string& frame)
{
     CLHEP::HepLorentzVector p_out=HectorParticle2LorentzVector(h_p,dir);
     LorentzBoost(p_out,frame);
     h_p=LorentzVector2HectorParticle(p_out);
}
void PPSTools::LorentzBoost(CLHEP::HepLorentzVector& p_out, const string& frame) 
{
    const long double microrad = 1.e-6;
    //
    double px_P, py_P,pz_P;
    double px_N, py_N,pz_N;
    double fBoostAngle1=0.;
    double fBoostAngle2=0.;
    if (frame=="LAB") {fBoostAngle1=fCrossingAngleBeam1;fBoostAngle2=fCrossingAngleBeam2;}
    if (frame=="MC")  {fBoostAngle1=-fCrossingAngleBeam1;fBoostAngle2=-fCrossingAngleBeam2;}
    px_P = fBeamMomentum*sin(fBoostAngle2*microrad);
    px_N = fBeamMomentum*sin(fBoostAngle1*microrad);
    pz_P = fBeamMomentum*cos(fBoostAngle2*microrad);
    pz_N = fBeamMomentum*cos(fBoostAngle1*microrad);
    py_P = 0.;
    py_N = 0.;

    CLHEP::HepLorentzVector BeamP, BeamN, projVect;
    BeamP.setPx(px_P);BeamP.setPy(py_P);BeamP.setPz(pz_P);BeamP.setE(fBeamEnergy);
    BeamN.setPx(px_N);BeamN.setPy(py_N);BeamN.setPz(-pz_N);BeamN.setE(fBeamEnergy);
    projVect = BeamP + BeamN;
    CLHEP::Hep3Vector beta;
    CLHEP::HepLorentzVector boosted = p_out;
    beta = projVect.boostVector();
    boosted.boost(beta);
    p_out=boosted;
}
void PPSTools::Get_t_and_xi(const CLHEP::HepLorentzVector* proton,double& t,double& xi) {
    t = 0.;
    xi = -1.;
    if (!proton) return;
    double mom = sqrt((proton->px())*(proton->px())+(proton->py())*(proton->py())+(proton->pz())*(proton->pz()));
    if (mom>fBeamMomentum) mom=fBeamMomentum;
    double energy = proton->e();
    double theta  = (proton->pz()>0)?proton->theta():CLHEP::pi-proton->theta();
    t      = -2.*(ProtonMassSQ-fBeamEnergy*energy+fBeamMomentum*mom*cos(theta));
    xi     = (1.0-energy/fBeamEnergy);
}
