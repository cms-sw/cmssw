#include "TopQuarkAnalysis/TopKinFitter/interface/TtFullLepKinSolver.h"
#include "TF2.h"

TtFullLepKinSolver::TtFullLepKinSolver():
  topmass_begin(0),
  topmass_end(0),
  topmass_step(0),
  mw(80.4),
  mb(4.8),
  pxmiss_(0),
  pymiss_(0)
{
  // That crude parametrisation has been obtained from a fit of O(1000) pythia events.
  // It is normalized to 1.
  EventShape_ = new TF2("landau2D","[0]*TMath::Landau(x,[1],[2],0)*TMath::Landau(y,[3],[4],0)",0,500,0,500);
  EventShape_->SetParameters(30.7137,56.2880,23.0744,59.1015,24.9145);
}

TtFullLepKinSolver::TtFullLepKinSolver(const double b, const double e, const double s, const std::vector<double>& nupars, const double mW, const double mB):
  topmass_begin(b),
  topmass_end(e),
  topmass_step(s),
  mw(mW),
  mb(mB),
  pxmiss_(0),
  pymiss_(0)
{
  EventShape_ = new TF2("landau2D","[0]*TMath::Landau(x,[1],[2],0)*TMath::Landau(y,[3],[4],0)",0,500,0,500);
  EventShape_->SetParameters(nupars[0],nupars[1],nupars[2],nupars[3],nupars[4]);
}

//
// destructor
//
TtFullLepKinSolver::~TtFullLepKinSolver() 
{
  delete EventShape_;
}

TtDilepEvtSolution TtFullLepKinSolver::addKinSolInfo(TtDilepEvtSolution * asol) 
{
  TtDilepEvtSolution fitsol(*asol);
  
  //antilepton and lepton
  TLorentzVector LV_e, LV_e_;
  //b and bbar quark
  TLorentzVector LV_b, LV_b_;
  
  bool hasMCinfo = true;
  if(fitsol.getGenN()) { // protect against non-dilept genevents
    genLV_n = TLorentzVector(fitsol.getGenN()->px(), fitsol.getGenN()->py(),
                             fitsol.getGenN()->pz(), fitsol.getGenN()->energy());
  } else hasMCinfo = false;

  if(fitsol.getGenNbar()) { // protect against non-dilept genevents
    genLV_n_ = TLorentzVector(fitsol.getGenNbar()->px(), fitsol.getGenNbar()->py(),
                              fitsol.getGenNbar()->pz(), fitsol.getGenNbar()->energy());
  } else hasMCinfo = false;
  // if MC is to be used to select the best top mass and is not available,
  // then nothing can be done. Stop here.
  if(useMCforBest_&&!hasMCinfo) return fitsol;

  // first lepton
  if (fitsol.getWpDecay() == "muon") {
    LV_e = TLorentzVector(fitsol.getMuonp().px(), fitsol.getMuonp().py(),
                          fitsol.getMuonp().pz(), fitsol.getMuonp().energy());
  } else if (fitsol.getWpDecay() == "electron") {
    LV_e = TLorentzVector(fitsol.getElectronp().px(), fitsol.getElectronp().py(),
                          fitsol.getElectronp().pz(), fitsol.getElectronp().energy());
  } else if (fitsol.getWpDecay() == "tau") {
    LV_e = TLorentzVector(fitsol.getTaup().px(), fitsol.getTaup().py(),
                          fitsol.getTaup().pz(), fitsol.getTaup().energy());
  }
    
  // second lepton
  if (fitsol.getWmDecay() == "muon") {
    LV_e_ = TLorentzVector(fitsol.getMuonm().px(), fitsol.getMuonm().py(),
                          fitsol.getMuonm().pz(), fitsol.getMuonm().energy());
  } else if (fitsol.getWmDecay() == "electron") {
    LV_e_ = TLorentzVector(fitsol.getElectronm().px(), fitsol.getElectronm().py(),
                           fitsol.getElectronm().pz(), fitsol.getElectronm().energy());
  } else if (fitsol.getWmDecay() == "tau") {
    LV_e_ = TLorentzVector(fitsol.getTaum().px(), fitsol.getTaum().py(),
                           fitsol.getTaum().pz(), fitsol.getTaum().energy());
  }

  // first jet
  LV_b = TLorentzVector(fitsol.getCalJetB().px(), fitsol.getCalJetB().py(),
                        fitsol.getCalJetB().pz(), fitsol.getCalJetB().energy());

  // second jet
  LV_b_ = TLorentzVector(fitsol.getCalJetBbar().px(), fitsol.getCalJetBbar().py(),
                         fitsol.getCalJetBbar().pz(), fitsol.getCalJetBbar().energy());
  
  //loop on top mass parameter
  double weightmax = -1e30;
  double mtmax = 0;
  for (double mt = topmass_begin; 
              mt < topmass_end + 0.5*topmass_step; 
              mt += topmass_step) {
    //cout << "mt = " << mt << endl;
    double q_coeff[5], q_sol[4];
    FindCoeff(LV_e, LV_e_, LV_b, LV_b_, mt, mt, pxmiss_, pymiss_, q_coeff);
    int NSol = quartic(q_coeff, q_sol);
    
    //loop on all solutions
    for (int isol = 0; isol < NSol; isol++) {
      TopRec(LV_e, LV_e_, LV_b, LV_b_, q_sol[isol]);
      double weight = useMCforBest_ ? WeightSolfromMC() : WeightSolfromShape();
      if (weight > weightmax) {
        weightmax =weight;
	mtmax = mt;
      }
    }
    
    //for (int i=0;i<5;i++) cout << " q_coeff["<<i<< "]= " << q_coeff[i];
    //cout << endl;
    
    //for (int i=0;i<4;i++) cout << " q_sol["<<i<< "]= " << q_sol[i];
    //cout << endl;
    //cout << "NSol_" << NSol << endl;
  }
  
  fitsol.setRecTopMass(mtmax);
  fitsol.setRecWeightMax(weightmax);
  
  return fitsol;
}

void
TtFullLepKinSolver::SetConstraints(const double xx, const double yy)
{
  pxmiss_ = xx;
  pymiss_ = yy;
}

TtFullLepKinSolver::NeutrinoSolution
TtFullLepKinSolver::getNuSolution(const TLorentzVector& LV_l, 
				  const TLorentzVector& LV_l_, 
				  const TLorentzVector& LV_b, 
				  const TLorentzVector& LV_b_)
{
  math::XYZTLorentzVector maxLV_n  = math::XYZTLorentzVector(0,0,0,0); 
  math::XYZTLorentzVector maxLV_n_ = math::XYZTLorentzVector(0,0,0,0);   

  //loop on top mass parameter
  double weightmax = -1;
  for(double mt = topmass_begin; 
             mt < topmass_end + 0.5*topmass_step; 
             mt += topmass_step) {
    double q_coeff[5], q_sol[4];
    FindCoeff(LV_l, LV_l_, LV_b, LV_b_, mt, mt, pxmiss_, pymiss_, q_coeff);
    int NSol = quartic(q_coeff, q_sol);
    
    //loop on all solutions
    for (int isol = 0; isol < NSol; isol++) {
      TopRec(LV_l, LV_l_, LV_b, LV_b_, q_sol[isol]);
      double weight = WeightSolfromShape();
      if (weight > weightmax) {
        weightmax =weight;
	maxLV_n.SetPxPyPzE(LV_n.Px(), LV_n.Py(), LV_n.Pz(), LV_n.E());
	maxLV_n_.SetPxPyPzE(LV_n_.Px(), LV_n_.Py(), LV_n_.Pz(), LV_n_.E());
      }
    }
  }
  TtFullLepKinSolver::NeutrinoSolution nuSol;
  nuSol.neutrino    = reco::LeafCandidate(0, maxLV_n  );
  nuSol.neutrinoBar = reco::LeafCandidate(0, maxLV_n_ ); 
  nuSol.weight = weightmax; 
  return nuSol;
}

void
TtFullLepKinSolver::FindCoeff(const TLorentzVector& al, 
			      const TLorentzVector& l,
			      const TLorentzVector& b_al,
			      const TLorentzVector& b_l,
			      const double mt, 
			      const double mat, 
			      const double px_miss, 
			      const double py_miss,
			      double* koeficienty)
{
  double E, apom1, apom2, apom3;
  double k11, k21, k31, k41,  cpom1, cpom2, cpom3, l11, l21, l31, l41, l51, l61, k1, k2, k3, k4, k5,k6;
  double l1, l2, l3, l4, l5, l6, k15, k25, k35, k45;

  C = -al.Px()-b_al.Px()-l.Px()-b_l.Px() + px_miss;
  D = -al.Py()-b_al.Py()-l.Py()-b_l.Py() + py_miss;

  // right side of first two linear equations - missing pT
  
  E = (sqr(mt)-sqr(mw)-sqr(mb))/(2*b_al.E())-sqr(mw)/(2*al.E())-al.E()+al.Px()*b_al.Px()/b_al.E()+al.Py()*b_al.Py()/b_al.E()+al.Pz()*b_al.Pz()/b_al.E();
  F = (sqr(mat)-sqr(mw)-sqr(mb))/(2*b_l.E())-sqr(mw)/(2*l.E())-l.E()+l.Px()*b_l.Px()/b_l.E()+l.Py()*b_l.Py()/b_l.E()+l.Pz()*b_l.Pz()/b_l.E();
  
  m1 = al.Px()/al.E()-b_al.Px()/b_al.E();
  m2 = al.Py()/al.E()-b_al.Py()/b_al.E();
  m3 = al.Pz()/al.E()-b_al.Pz()/b_al.E();
  
  n1 = l.Px()/l.E()-b_l.Px()/b_l.E();
  n2 = l.Py()/l.E()-b_l.Py()/b_l.E();
  n3 = l.Pz()/l.E()-b_l.Pz()/b_l.E();
  
  pom = E-m1*C-m2*D;
  apom1 = sqr(al.Px())-sqr(al.E());
  apom2 = sqr(al.Py())-sqr(al.E());
  apom3 = sqr(al.Pz())-sqr(al.E());
  
  k11 = 1/sqr(al.E())*(pow(mw,4)/4+sqr(C)*apom1+sqr(D)*apom2+apom3*sqr(pom)/sqr(m3)+sqr(mw)*(al.Px()*C+al.Py()*D+al.Pz()*pom/m3)+2*al.Px()*al.Py()*C*D+2*al.Px()*al.Pz()*C*pom/m3+2*al.Py()*al.Pz()*D*pom/m3);
  k21 = 1/sqr(al.E())*(-2*C*m3*n3*apom1+2*apom3*n3*m1*pom/m3-sqr(mw)*m3*n3*al.Px()+sqr(mw)*m1*n3*al.Pz()-2*al.Px()*al.Py()*D*m3*n3+2*al.Px()*al.Pz()*C*m1*n3-2*al.Px()*al.Pz()*n3*pom+2*al.Py()*al.Pz()*D*m1*n3);
  k31 = 1/sqr(al.E())*(-2*D*m3*n3*apom2+2*apom3*n3*m2*pom/m3-sqr(mw)*m3*n3*al.Py()+sqr(mw)*m2*n3*al.Pz()-2*al.Px()*al.Py()*C*m3*n3+2*al.Px()*al.Pz()*C*m2*n3-2*al.Py()*al.Pz()*n3*pom+2*al.Py()*al.Pz()*D*m2*n3);
  k41 = 1/sqr(al.E())*(2*apom3*m1*m2*sqr(n3)+2*al.Px()*al.Py()*sqr(m3)*sqr(n3)-2*al.Px()*al.Pz()*m2*m3*sqr(n3)-2*al.Py()*al.Pz()*m1*m3*sqr(n3));
  k51 = 1/sqr(al.E())*(apom1*sqr(m3)*sqr(n3)+apom3*sqr(m1)*sqr(n3)-2*al.Px()*al.Pz()*m1*m3*sqr(n3));
  k61 = 1/sqr(al.E())*(apom2*sqr(m3)*sqr(n3)+apom3*sqr(m2)*sqr(n3)-2*al.Py()*al.Pz()*m2*m3*sqr(n3));
  
  cpom1 = sqr(l.Px())-sqr(l.E());
  cpom2 = sqr(l.Py())-sqr(l.E());
  cpom3 = sqr(l.Pz())-sqr(l.E());
  
  l11 = 1/sqr(l.E())*(pow(mw,4)/4+cpom3*sqr(F)/sqr(n3)+sqr(mw)*l.Pz()*F/n3);
  l21 = 1/sqr(l.E())*(-2*cpom3*F*m3*n1/n3+sqr(mw)*(l.Px()*m3*n3-l.Pz()*n1*m3)+2*l.Px()*l.Pz()*F*m3);
  l31 = 1/sqr(l.E())*(-2*cpom3*F*m3*n2/n3+sqr(mw)*(l.Py()*m3*n3-l.Pz()*n2*m3)+2*l.Py()*l.Pz()*F*m3);
  l41 = 1/sqr(l.E())*(2*cpom3*n1*n2*sqr(m3)+2*l.Px()*l.Py()*sqr(m3)*sqr(n3)-2*l.Px()*l.Pz()*n2*n3*sqr(m3)-2*l.Py()*l.Pz()*n1*n3*sqr(m3));
  l51 = 1/sqr(l.E())*(cpom1*sqr(m3)*sqr(n3)+cpom3*sqr(n1)*sqr(m3)-2*l.Px()*l.Pz()*n1*n3*sqr(m3));
  l61 = 1/sqr(l.E())*(cpom2*sqr(m3)*sqr(n3)+cpom3*sqr(n2)*sqr(m3)-2*l.Py()*l.Pz()*n2*n3*sqr(m3));
  
  k1 = k11*k61;
  k2 = k61*k21/k51;
  k3 = k31;
  k4 = k41/k51;
  k5 = k61/k51;
  k6 = 1;
  
  l1 = l11*k61;
  l2 = l21*k61/k51;
  l3 = l31;
  l4 = l41/k51;
  l5 = l51*k61/(sqr(k51));
  l6 = l61/k61;
  
  k15 = k1*l5-l1*k5;
  k25 = k2*l5-l2*k5;
  k35 = k3*l5-l3*k5;
  k45 = k4*l5-l4*k5;
  
  k16 = k1*l6-l1*k6;
  k26 = k2*l6-l2*k6;
  k36 = k3*l6-l3*k6;
  k46 = k4*l6-l4*k6;
  k56 = k5*l6-l5*k6;

  koeficienty[0] = k15*sqr(k36)-k35*k36*k16-k56*sqr(k16);
  koeficienty[1] = 2*k15*k36*k46+k25*sqr(k36)+k35*(-k46*k16-k36*k26)-k45*k36*k16-2*k56*k26*k16;
  koeficienty[2] = k15*sqr(k46)+2*k25*k36*k46+k35*(-k46*k26-k36*k56)-k56*(sqr(k26)+2*k56*k16)-k45*(k46*k16+k36*k26);
  koeficienty[3] = k25*sqr(k46)-k35*k46*k56-k45*(k46*k26+k36*k56)-2*sqr(k56)*k26;
  koeficienty[4] = -k45*k46*k56-pow(k56,3);
  
  // normalization of coefficients
  int moc=(int(log10(fabs(koeficienty[0])))+int(log10(fabs(koeficienty[4]))))/2;
  
  koeficienty[0]=koeficienty[0]/TMath::Power(10,moc);
  koeficienty[1]=koeficienty[1]/TMath::Power(10,moc);
  koeficienty[2]=koeficienty[2]/TMath::Power(10,moc);
  koeficienty[3]=koeficienty[3]/TMath::Power(10,moc);
  koeficienty[4]=koeficienty[4]/TMath::Power(10,moc);
}

void TtFullLepKinSolver::TopRec(const TLorentzVector& al, 
                                const TLorentzVector& l,
	                        const TLorentzVector& b_al,
	                        const TLorentzVector& b_l, 
				const double sol)
{
  TVector3 t_ttboost;
  TLorentzVector aux;
  double pxp, pyp, pzp, pup, pvp, pwp;
    
  pxp = sol*(m3*n3/k51);   
  pyp = -(m3*n3/k61)*(k56*pow(sol,2) + k26*sol + k16)/(k36 + k46*sol);
  pzp = -1/n3*(n1*pxp + n2*pyp - F);
  pwp = 1/m3*(m1*pxp + m2*pyp + pom);
  pup = C - pxp;
  pvp = D - pyp;
     
  LV_n_.SetXYZM(pxp, pyp, pzp, 0.0);
  LV_n.SetXYZM(pup, pvp, pwp, 0.0);
  
  LV_t_ = b_l + l + LV_n_;
  LV_t = b_al + al + LV_n;  
 
  aux = (LV_t_ + LV_t);
  t_ttboost = -aux.BoostVector();
  LV_tt_t_ = LV_t_;
  LV_tt_t = LV_t;
  LV_tt_t_.Boost(t_ttboost);
  LV_tt_t.Boost(t_ttboost); 
}

double
TtFullLepKinSolver::WeightSolfromMC() const
{
  double weight = 1;
  weight = ((LV_n.E() > genLV_n.E())? genLV_n.E()/LV_n.E(): LV_n.E()/genLV_n.E())
           *((LV_n_.E() > genLV_n_.E())? genLV_n_.E()/LV_n_.E(): LV_n_.E()/genLV_n_.E());
  return weight;
}

double
TtFullLepKinSolver::WeightSolfromShape() const
{
  return EventShape_->Eval(LV_n.E(),LV_n_.E());
}
		     
int
TtFullLepKinSolver::quartic(double *koeficienty, double* koreny) const
{
  double w, b0, b1, b2;
  double c[4];
  double d0, d1, h, t, z;
  double *px;
 
  if (koeficienty[4]==0.0) 
    return cubic(koeficienty, koreny);
  /* quartic problem? */
  w = koeficienty[3]/(4*koeficienty[4]);
  /* offset */
  b2 = -6*sqr(w) + koeficienty[2]/koeficienty[4];
  /* koeficienty. of shifted polynomial */
  b1 = (8*sqr(w) - 2*koeficienty[2]/koeficienty[4])*w + koeficienty[1]/koeficienty[4];
  b0 = ((-3*sqr(w) + koeficienty[2]/koeficienty[4])*w - koeficienty[1]/koeficienty[4])*w + koeficienty[0]/koeficienty[4];

  c[3] = 1.0;
  /* cubic resolvent */
  c[2] = b2;
  c[1] = -4*b0;
  c[0] = sqr(b1) - 4*b0*b2;
  
  cubic(c, koreny);
  z = koreny[0];
  //double z1=1.0,z2=2.0,z3=3.0;
  //TMath::RootsCubic(c,z1,z2,z3);
  //if (z2 !=0) z = z2;
  //if (z1 !=0) z = z1;
  /* only lowermost root needed */

  int nreal = 0;
  px = koreny;
  t = sqrt(0.25*sqr(z) - b0);
  for(int i=-1; i<=1; i+=2) {
    d0 = -0.5*z + i*t;
    /* coeffs. of quadratic factor */
    d1 = (t!=0.0)? -i*0.5*b1/t : i*sqrt(-z - b2);
    h = 0.25*sqr(d1) - d0;
    if (h>=0.0) {
      h = sqrt(h);
      nreal += 2;
      *px++ = -0.5*d1 - h - w;
      *px++ = -0.5*d1 + h - w;
    }
  }

  //  if (nreal==4) {
    /* sort results */
//    if (koreny[2]<koreny[0]) SWAP(koreny[0], koreny[2]);
//    if (koreny[3]<koreny[1]) SWAP(koreny[1], koreny[3]);
//    if (koreny[1]<koreny[0]) SWAP(koreny[0], koreny[1]);
//    if (koreny[3]<koreny[2]) SWAP(koreny[2], koreny[3]);
//    if (koreny[2]<koreny[1]) SWAP(koreny[1], koreny[2]);
//  }
  return nreal;

}

int
TtFullLepKinSolver::cubic(const double *coeffs, double* koreny) const
{
  unsigned nreal;
  double w, p, q, dis, h, phi;
  
  if (coeffs[3]!=0.0) {
    /* cubic problem? */
    w = coeffs[2]/(3*coeffs[3]);
    p = sqr(coeffs[1]/(3*coeffs[3])-sqr(w))*(coeffs[1]/(3*coeffs[3])-sqr(w));
    q = -0.5*(2*sqr(w)*w-(coeffs[1]*w-coeffs[0])/coeffs[3]);
    dis = sqr(q)+p;
    /* discriminant */
    if (dis<0.0) {
      /* 3 real solutions */
      h = q/sqrt(-p);
      if (h>1.0) h = 1.0;
      /* confine the argument of */
      if (h<-1.0) h = -1.0;
      /* acos to [-1;+1] */
      phi = acos(h);
      p = 2*TMath::Power(-p, 1.0/6.0);
      for(unsigned i=0; i<3; i++) 
	koreny[i] = p*cos((phi+2*i*TMath::Pi())/3.0) - w;
      if (koreny[1]<koreny[0]) SWAP(koreny[0], koreny[1]);
      /* sort results */
      if (koreny[2]<koreny[1]) SWAP(koreny[1], koreny[2]);
      if (koreny[1]<koreny[0]) SWAP(koreny[0], koreny[1]);
      nreal = 3;
    }
    else {
      /* only one real solution */
      dis = sqrt(dis);
      h = TMath::Power(fabs(q+dis), 1.0/3.0);
      p = TMath::Power(fabs(q-dis), 1.0/3.0);
      koreny[0] = ((q+dis>0.0)? h : -h) + ((q-dis>0.0)? p : -p) -  w;
      nreal = 1;
    }

    /* Perform one step of a Newton iteration in order to minimize
       round-off errors */
    for(unsigned i=0; i<nreal; i++) {
      h = coeffs[1] + koreny[i] * (2 * coeffs[2] + 3 * koreny[i] * coeffs[3]);
      if (h != 0.0)
	koreny[i] -= (coeffs[0] + koreny[i] * (coeffs[1] + koreny[i] * (coeffs[2] + koreny[i] * coeffs[3])))/h;
    }
  }

  else if (coeffs[2]!=0.0) {
    /* quadratic problem? */
    p = 0.5*coeffs[1]/coeffs[2];
    dis = sqr(p) - coeffs[0]/coeffs[2];
    if (dis>=0.0) {
      /* two real solutions */
      dis = sqrt(dis);
      koreny[0] = -p - dis;
      koreny[1] = -p + dis;
      nreal = 2;
    }
    else
      /* no real solution */
      nreal = 0;
  }

  else if (coeffs[1]!=0.0) {
    /* linear problem? */
    koreny[0] = -coeffs[0]/coeffs[1];
    nreal = 1;
  }

  else
    /* no equation */
    nreal = 0;
  
  return nreal;
}


void
TtFullLepKinSolver::SWAP(double& realone, double& realtwo) const
{
  if (realtwo < realone) {
    double aux = realtwo;
    realtwo = realone;
    realone = aux;
  }
}
