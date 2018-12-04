#include "Utilities/PPS/interface/PPSUtilities.h"
#include "HepMC/GenParticle.h"
#include "H_BeamParticle.h"
#include "TLorentzVector.h"

TLorentzVector PPSTools::HectorParticle2LorentzVector(H_BeamParticle hp,int direction)
{
     double partP = sqrt(pow(hp.getE(),2)-ProtonMassSQ);
     double theta = sqrt(pow(hp.getTX(),2)+pow(hp.getTY(),2))*urad;
     double pz = partP*cos(theta);
     double px = tan((double)hp.getTX()*urad)*pz;//it is equivalente to PartP*sin(theta)*cos(phi);
     double py = tan((double)hp.getTY()*urad)*pz;//it is equivalente to partP*sin(theta)*sin(phi);
     pz*=direction;
     return TLorentzVector(px,py,pz,hp.getE());
}

H_BeamParticle PPSTools::LorentzVector2HectorParticle(TLorentzVector p)
{
     H_BeamParticle h_p;
     h_p.set4Momentum(p.Px(),p.Py(),abs(p.Pz()),p.E());
     return h_p;
}
void PPSTools::LorentzBoost(HepMC::GenParticle& p, const string& frame)
{
     TLorentzVector p_out;
     p_out.SetPx(p.momentum().px());
     p_out.SetPy(p.momentum().py());
     p_out.SetPz(p.momentum().pz());
     LorentzBoost(p_out,frame);
     p.set_momentum(HepMC::FourVector(p_out.Px(),p_out.Py(),p_out.Pz(),p_out.E()));
}
void PPSTools::LorentzBoost(H_BeamParticle& h_p, int dir , const string& frame)
{
     TLorentzVector p_out=HectorParticle2LorentzVector(h_p,dir);
     LorentzBoost(p_out,frame);
     h_p=LorentzVector2HectorParticle(p_out);
}
void PPSTools::LorentzBoost(TLorentzVector& p_out, const string& frame)
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

    TLorentzVector BeamP, BeamN, projVect;
    BeamP.SetPx(px_P);BeamP.SetPy(py_P);BeamP.SetPz(pz_P);BeamP.SetE(fBeamEnergy);
    BeamN.SetPx(px_N);BeamN.SetPy(py_N);BeamN.SetPz(-pz_N);BeamN.SetE(fBeamEnergy);
    projVect = BeamP + BeamN;
    TVector3 beta;
    TLorentzVector boosted = p_out;
    beta = projVect.BoostVector();
    boosted.Boost(beta);
    p_out=boosted;
}
void PPSTools::Get_t_and_xi(const TLorentzVector* proton,double& t,double& xi) {
    t = 0.;
    xi = -1.;
    if (!proton) return;
    double mom = sqrt((proton->Px())*(proton->Px())+(proton->Py())*(proton->Py())+(proton->Pz())*(proton->Pz()));
    if (mom>fBeamMomentum) mom=fBeamMomentum;
    double energy = proton->E();
    double theta  = (proton->Pz()>0)?proton->Theta():CLHEP::pi-proton->Theta();
    t      = -2.*(ProtonMassSQ-fBeamEnergy*energy+fBeamMomentum*mom*cos(theta));
    xi     = (1.0-energy/fBeamEnergy);
}
