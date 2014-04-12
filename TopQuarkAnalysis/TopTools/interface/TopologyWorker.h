// -*- C++ -*-
//
// Package:    TopTools
// Class:      TopologyWorker
// 
/**\class TopologyWorker TopologyWorker.cc TopQuarkAnalysis/TopTools/interface/TopologyWorker.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
     This class contains the topological methods as used in D0 (all hadronic) analyses.
*/
#ifndef __TOPTOOLSTOPOLOGYWORKER__
#define __TOPTOOLSTOPOLOGYWORKER__

#warning The TopologyWorker class is currently not supported anymore! There might be issues in its implementation.
#warning If you are still using it or planned to do so, please contact the admins of the corresponding CMSSW package.
#warning You can find their e-mail adresses in: cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/TopQuarkAnalysis/TopTools/.admin/

#include "TF1.h"
#include "TMath.h"
#include "TClass.h"
#include "TString.h"
#include "TRandom.h"
#include "TMatrixD.h"
#include "TLorentzVector.h"

#include <cmath>
#include <iostream>

class TopologyWorker
{
public:
  TopologyWorker(){;}
  TopologyWorker(bool boost);
  virtual ~TopologyWorker(){;}

  void clear(void){m_np=0;m_np2=0;return;}
  
  void setPartList(TObjArray* e1, TObjArray* e2);
  void setVerbose(bool loud){m_verbose=loud; return;}
  
  void     setThMomPower(double tp);
  double   getThMomPower();
  void     setFast(int nf);
  int      getFast();
  
  TVector3 thrustAxis();
  TVector3 majorAxis();
  TVector3 minorAxis();
  
  TVector3 thrust(); 
  // thrust :: Corresponding thrust, major, and minor value.
  
  double oblateness();
  double get_sphericity();
  double get_aplanarity();
  double get_h10();
  double get_h20();
  double get_h30();
  double get_h40();
  double get_h50();
  double get_h60();


  void planes_sphe(double& pnorm,double& p2, double& p3);
  void planes_sphe_wei(double& pnorm,double& p2, double& p3);
  void planes_thrust(double& pnorm,double& p2, double& p3);
  void sumangles(float& sdeta, float& sdr);
  
  double get_ht() {return m_ht;}
  double get_ht3() {return m_ht3;}
  double get_et0() {return m_et0;}
  double get_sqrts() {return m_sqrts;}
  double get_njetW() {return m_njetsweighed;}
  double get_et56() {return m_et56;}
  double get_centrality() { return m_centrality;}
  
private:	
  bool m_verbose;
  void getetaphi(double px, double py, double pz, double& eta, double& phi);
  double ulAngle(double x, double y);
  double sign(double a, double b);
  void     ludbrb(TMatrix *mom, 
		  double the, 
		  double phi, 
		  double bx, 
		  double by,
		  double bz);
  
  int iPow(int man, int exp);
  
  double m_dSphMomPower; 
  // PARU(41): Power of momentum dependence in sphericity finder.
  
  double m_dDeltaThPower;
  // PARU(42): Power of momentum dependence in thrust finder.	
  
  int m_iFast; 
  // MSTU(44): # of initial fastest particles choosen to start search.
  
  double m_dConv;
  // PARU(48): Convergence criteria for axis maximization.
  
  int m_iGood;
  // MSTU(45): # different starting configurations that must
  // converge before axis is accepted as correct.	
  
  TMatrix m_dAxes;
  // data: results
  // m_dAxes[1] is the Thrust axis.
  // m_dAxes[2] is the Major axis.
  // m_dAxes[3] is the Minor axis.
  
  TVector3 m_ThrustAxis;
  TVector3 m_MajorAxis;
  TVector3 m_MinorAxis;
  TVector3 m_Thrust;
  
  TRandom m_random;
  
  TMatrix m_mom;
  TMatrix m_mom2;
  
  double m_dThrust[4];
  double m_dOblateness;
  int m_np;
  int m_np2;
  bool m_sanda_called;
  bool m_fowo_called;
  bool m_boost;
  bool m_sumangles_called;
  double m_sph;
  double m_apl;
  double m_h10;
  double m_h20;
  double m_h30;
  double m_h40;
  double m_h50;
  double m_h60;
  double m_ht;
  double m_ht3;
  double m_et0;
  double m_sqrts;
  double m_njetsweighed;
  double m_et56;
  double m_centrality;
  
  void sanda();
  void fowo();
  static int m_maxpart;
  
  void CalcWmul();
  void CalcSqrts();
  void CalcHTstuff();
  double CalcPt(int i) { return sqrt(pow(m_mom(i,1),2)+pow(m_mom(i,2),2));}
  double CalcPt2(int i)  { return sqrt(pow(m_mom2(i,1),2)+pow(m_mom2(i,2),2));}
  double CalcEta(int i) {double eta, phi;getetaphi(m_mom(i,1),m_mom(i,2),m_mom(i,3),eta,phi); return eta;}
  double CalcEta2(int i) {double eta, phi; getetaphi(m_mom2(i,1),m_mom2(i,2),m_mom2(i,3),eta,phi); return eta;}
  
};

class LessThan {
   public :
     // retrieve tru info MC stuff
  bool operator () (const TLorentzVector & tl1, const TLorentzVector &
		    tl2)
    const {
    return tl2.Pt() < tl1.Pt();
  }
};

Int_t TopologyWorker::m_maxpart = 1000;

TopologyWorker::TopologyWorker(bool boost):
  m_dSphMomPower(2.0),m_dDeltaThPower(0),
  m_iFast(4),m_dConv(0.0001),m_iGood(2)
{
  m_dAxes.ResizeTo(4,4);
  m_mom.ResizeTo(m_maxpart,6);
  m_mom2.ResizeTo(m_maxpart,6);
  m_np=-1;
  m_np2=-1;
  m_sanda_called=false;
  m_fowo_called=false;
  m_sumangles_called=false;
  m_verbose=false;
  m_boost=boost;
  m_sph=-1;
  m_apl=-1;
  m_h10=-1;
  m_h20=-1;
  m_h30=-1;
  m_h40=-1;
  m_sqrts=0;
  m_ht=0;
  m_ht3=0;
  m_et56=0;
  m_njetsweighed=0;
  m_et0=0;
}
//______________________________________________________________


// Input the particle 3(4)-vector list
// e: 3-vector  TVector3       ..(px,py,pz) or
//    4-vector  TLorentzVector ..(px,py,pz,E) 
// Even input the TLorentzVector, we don't use Energy 
// 
void TopologyWorker::setPartList(TObjArray* e1, TObjArray* e2)
{	
  //To make this look like normal physics notation the
  //zeroth element of each array, mom[i][0], will be ignored
  //and operations will be on elements 1,2,3...
  TMatrix mom(m_maxpart,6);
  TMatrix mom2(m_maxpart,6);
  double tmax = 0;
  double phi = 0.;
  double the = 0.;
  double sgn;
  TMatrix fast(m_iFast + 1,6);
  TMatrix work(11,6);
  double tdi[4] = {0.,0.,0.,0.};
  double tds;
  double tpr[4] = {0.,0.,0.,0.};
  double thp;
  double thps;
  double pxtot=0;
  double pytot=0;
  double pztot=0;
  double etot=0;

  TMatrix temp(3,5);
  Int_t np = 0;
  Int_t numElements = e1->GetEntries();
  Int_t numElements2 = e2->GetEntries();

  // trying to sort...
  
  

  m_np=0;
  for(Int_t elem=0;elem<numElements;elem++) {
    if(m_verbose){
      std::cout << "looping over array, element " << elem << std::endl;
    }
    TObject* o = (TObject*) e1->At(elem);    
    if(m_verbose){
      std::cerr << "TopologyWorker:SetPartList(): adding jet " << elem  << "." << std::endl; 
    }
    if (np >= m_maxpart) { 
	printf("Too many particles input to TopologyWorker");
	return;
    }
    if(m_verbose){
      std::cout << "getting name of object..." << std::endl;
    }
    TString nam(o->IsA()->GetName());
    if(m_verbose){
      std::cout << "TopologyWorker::setPartList(): object is of type " << nam << std::endl;
    }
    if (nam.Contains("TVector3")) {
      TVector3 v(((TVector3 *) o)->X(),
		 ((TVector3 *) o)->Y(),
		 ((TVector3 *) o)->Z());
      mom(np,1) = v.X();
      mom(np,2) = v.Y();
      mom(np,3) = v.Z();
      mom(np,4) = v.Mag();
    }
    else if (nam.Contains("TLorentzVector")) {
      TVector3 v(((TLorentzVector *) o)->X(),
		 ((TLorentzVector *) o)->Y(),
		 ((TLorentzVector *) o)->Z());
      mom(np,1) = v.X();
      mom(np,2) = v.Y();
      mom(np,3) = v.Z();
      mom(np,4) = ((TLorentzVector *) o)->T();
    }
    else {
      printf("TopologyWorker::setEvent input is not a TVector3 or a TLorentzVector\n");
      continue;
    }
    

    if ( TMath::Abs( m_dDeltaThPower ) <= 0.001 ) {
      mom(np,5) = 1.0;
    }
    else {
      mom(np,5) = TMath::Power(mom(np,4),m_dDeltaThPower);
    }
    tmax = tmax + mom(np,4)*mom(np,5);
    pxtot+=mom(np,1);
    pytot+=mom(np,2);
    pztot+=mom(np,3);
    etot+=mom(np,4);
    np++;
    m_np=np;
  }
  Int_t np2=0;
  // second jet array.... only use some values here.
  for(Int_t elem=0;elem<numElements2;elem++) {
    //cout << elem << endl;
    TObject* o = (TObject*) e2->At(elem);    
    if (np2 >= m_maxpart) { 
	printf("Too many particles input to TopologyWorker");
	return;
    }

    TString nam(o->IsA()->GetName());
    if (nam.Contains("TVector3")) {
      TVector3 v(((TVector3 *) o)->X(),
		 ((TVector3 *) o)->Y(),
		 ((TVector3 *) o)->Z());
      mom2(np2,1) = v.X();
      mom2(np2,2) = v.Y();
      mom2(np2,3) = v.Z();
      mom2(np2,4) = v.Mag();
    }
    else if (nam.Contains("TLorentzVector")) {
      TVector3 v(((TLorentzVector *) o)->X(),
		 ((TLorentzVector *) o)->Y(),
		 ((TLorentzVector *) o)->Z());
      mom2(np2,1) = v.X();
      mom2(np2,2) = v.Y();
      mom2(np2,3) = v.Z();
      mom2(np2,4) = ((TLorentzVector *) o)->T();
      //      cout << "mom2: " << mom2(np2,1) << ", " << mom2(np2,2)<<", " << mom2(np2,3)<<", " << mom2(np2,4)<< endl;
    }
    else {
      printf("TopologyWorker::setEvent Second vector input is not a TVector3 or a TLorentzVector\n");
      continue;
    }
    np2++;
    m_np2=np2;
  }
  m_mom2=mom2;

  if (m_boost && m_np>1) {
    printf("TopologyWorker::setEvent Only boosting first vector so watch out when you do this!!!");
    TVector3 booze(-pxtot/etot,-pytot/etot,-pztot/etot);
    TLorentzVector l1;
    for (int k=0; k<m_np ; k++) {
      l1.SetPx(mom(k,1));
      l1.SetPy(mom(k,2)); 
      l1.SetPz(mom(k,3));
      l1.SetE(mom(k,4));
      l1.Boost(booze);
      mom(k,1)=l1.Px();
      mom(k,2)=l1.Py();
      mom(k,3)=l1.Pz();
      mom(k,4)=l1.E();
    }
  }
  
  m_sanda_called=false; 
  m_fowo_called=false; 
  for (int ip=0;ip<m_maxpart;ip++) {
    for (int id=0;id<6;id++) {
      m_mom(ip,id)=mom(ip,id);
    }
  }

  if ( np < 2 ) {
    m_dThrust[1] = -1.0;
    m_dOblateness = -1.0;
    return;
  }
  // for pass = 1: find thrust axis.
  // for pass = 2: find major axis.
  for ( Int_t pass=1; pass < 3; pass++) {
    if ( pass == 2 ) {
      phi = ulAngle(m_dAxes(1,1), m_dAxes(1,2));
      ludbrb( &mom, 0, -phi, 0., 0., 0. );
      for ( Int_t i = 0; i < 3; i++ ) {
	for ( Int_t j = 1; j < 4; j++ ) {
	  temp(i,j) = m_dAxes(i+1,j);
	}
	temp(i,4) = 0;
      }
      ludbrb(&temp,0.,-phi,0.,0.,0.);
      for ( Int_t ib = 0; ib < 3; ib++ ) {
	for ( Int_t j = 1; j < 4; j++ ) {
	  m_dAxes(ib+1,j) = temp(ib,j);
	}
      }
      the = ulAngle( m_dAxes(1,3), m_dAxes(1,1) );
      ludbrb( &mom, -the, 0., 0., 0., 0. );
      for ( Int_t ic = 0; ic < 3; ic++ ) {
	for ( Int_t j = 1; j < 4; j++ ) {
	  temp(ic,j) = m_dAxes(ic+1,j);
	}
	temp(ic,4) = 0;
      }
      ludbrb(&temp,-the,0.,0.,0.,0.);
      for ( Int_t id = 0; id < 3; id++ ) {	
	for ( Int_t j = 1; j < 4; j++ ) {
	  m_dAxes(id+1,j) = temp(id,j);
	}
      }
    }
    for ( Int_t ifas = 0; ifas < m_iFast + 1 ; ifas++ ) {
      fast(ifas,4) = 0.;
    }
    // Find the m_iFast highest momentum particles and
    // put the highest in fast[0], next in fast[1],....fast[m_iFast-1].
    // fast[m_iFast] is just a workspace.
    for ( Int_t i = 0; i < np; i++ ) {
      if ( pass == 2 ) {
	mom(i,4) = TMath::Sqrt( mom(i,1)*mom(i,1) 
			      + mom(i,2)*mom(i,2) ); 
      }
      for ( Int_t ifas = m_iFast - 1; ifas > -1; ifas-- ) {
	if ( mom(i,4) > fast(ifas,4) ) {
	  for ( Int_t j = 1; j < 6; j++ ) {
	    fast(ifas+1,j) = fast(ifas,j);
	    if ( ifas == 0 ) fast(ifas,j) = mom(i,j);	    
	  }
	}
	else {
	  for ( Int_t j = 1; j < 6; j++ ) {
	    fast(ifas+1,j) = mom(i,j);
	  }
	  break;
	}
      }
    }
    // Find axis with highest thrust (case 1)/ highest major (case 2).
    for ( Int_t ie = 0; ie < work.GetNrows(); ie++ ) {
      work(ie,4) = 0.;
    }
    Int_t p = TMath::Min( m_iFast, np ) - 1;
    // Don't trust Math.pow to give right answer always.
    // Want nc = 2**p.
    Int_t nc = iPow(2,p); 
    for ( Int_t n = 0; n < nc; n++ ) {
      for ( Int_t j = 1; j < 4; j++ ) {
	tdi[j] = 0.;
      }
      for ( Int_t i = 0; i < TMath::Min(m_iFast,n); i++ ) {
	sgn = fast(i,5);
	if (iPow(2,(i+1))*((n+iPow(2,i))/iPow(2,(i+1))) >= i+1){
	  sgn = -sgn;
	}
	for ( Int_t j = 1; j < 5-pass; j++ ) {
	  tdi[j] = tdi[j] + sgn*fast(i,j);
	}
      }
      tds = tdi[1]*tdi[1] + tdi[2]*tdi[2] + tdi[3]*tdi[3];
      for ( Int_t iw = TMath::Min(n,9); iw > -1; iw--) {
	if( tds > work(iw,4) ) {
	  for ( Int_t j = 1; j < 5; j++ ) {
	    work(iw+1,j) = work(iw,j);
	    if ( iw == 0 ) {
	      if ( j < 4 ) {
		work(iw,j) = tdi[j];
	      }
	      else {
		work(iw,j) = tds;
	      }
	    }
	  }
	}
	else {
	  for ( Int_t j = 1; j < 4; j++ ) {
	    work(iw+1,j) = tdi[j];
	  }
	  work(iw+1,4) = tds;
	}
      }
    }
    // Iterate direction of axis until stable maximum.
    m_dThrust[pass] = 0;
    thp = -99999.;
    Int_t nagree = 0;
    for ( Int_t iw = 0; 
	  iw < TMath::Min(nc,10) && nagree < m_iGood; iw++ ){
      thp = 0.;
      thps = -99999.;
      while ( thp > thps + m_dConv ) {
	thps = thp;
	for ( Int_t j = 1; j < 4; j++ ) {
	  if ( thp <= 1E-10 ) {
	    tdi[j] = work(iw,j);
	  }
	  else {
	    tdi[j] = tpr[j];
	    tpr[j] = 0;
	  }
	}
	for ( Int_t i = 0; i < np; i++ ) {
	  sgn = sign(mom(i,5), 
		     tdi[1]*mom(i,1) + 
		     tdi[2]*mom(i,2) + 
		     tdi[3]*mom(i,3));
	  for ( Int_t j = 1; j < 5 - pass; j++ ){
	    tpr[j] = tpr[j] + sgn*mom(i,j);
	  }
	}
	thp = TMath::Sqrt(tpr[1]*tpr[1] 
			  + tpr[2]*tpr[2] 
			  + tpr[3]*tpr[3])/tmax;
      }
      // Save good axis. Try new initial axis until enough
      // tries agree.
      if ( thp < m_dThrust[pass] - m_dConv ) {
	break;
      }
      if ( thp > m_dThrust[pass] + m_dConv ) {
	nagree = 0;
	sgn = iPow( -1, (Int_t)TMath::Nint(m_random.Rndm()) );
	for ( Int_t j = 1; j < 4; j++ ) {
	  m_dAxes(pass,j) = sgn*tpr[j]/(tmax*thp);
	}
	m_dThrust[pass] = thp;
      }
      nagree = nagree + 1;
    }
  }
  // Find minor axis and value by orthogonality.
  sgn = iPow( -1, (Int_t)TMath::Nint(m_random.Rndm()));
  m_dAxes(3,1) = -sgn*m_dAxes(2,2);
  m_dAxes(3,2) = sgn*m_dAxes(2,1);
  m_dAxes(3,3) = 0.;
  thp = 0.;
  for ( Int_t i = 0; i < np; i++ ) {
    thp += mom(i,5)*TMath::Abs(m_dAxes(3,1)*mom(i,1) + 
			       m_dAxes(3,2)*mom(i,2));
  }
  m_dThrust[3] = thp/tmax;
  // Rotate back to original coordinate system.
  for ( Int_t i6 = 0; i6 < 3; i6++ ) {
    for ( Int_t j = 1; j < 4; j++ ) {
      temp(i6,j) = m_dAxes(i6+1,j);
    }
    temp(i6,4) = 0;
  }
  ludbrb(&temp,the,phi,0.,0.,0.);
  for ( Int_t i7 = 0; i7 < 3; i7++ ) {
    for ( Int_t j = 1; j < 4; j++ ) {
      m_dAxes(i7+1,j) = temp(i7,j);
    }
  }
  m_dOblateness = m_dThrust[2] - m_dThrust[3];
  
  // more stuff:
  
  // calculate weighed jet multiplicity();
  CalcWmul();
  CalcHTstuff();
  CalcSqrts();
 
}
//______________________________________________________________
	
// Setting and getting parameters.
	
void TopologyWorker::setThMomPower(double tp)
{
  // Error if sp not positive.
  if ( tp > 0. ) m_dDeltaThPower = tp - 1.0;
  return;
}
//______________________________________________________________

double TopologyWorker::getThMomPower()
{
  return 1.0 + m_dDeltaThPower;
}
//______________________________________________________________

void TopologyWorker::setFast(Int_t nf)
{
  // Error if sp not positive.
  if ( nf > 3 ) m_iFast = nf;
  return;
}
//______________________________________________________________

Int_t TopologyWorker::getFast()
{
  return m_iFast;
}
//______________________________________________________________

// Returning results

TVector3 TopologyWorker::thrustAxis() {
  TVector3 m_ThrustAxis(m_dAxes(1,1),m_dAxes(1,2),m_dAxes(1,3));
  return m_ThrustAxis;
}
//______________________________________________________________

TVector3 TopologyWorker::majorAxis() {
  TVector3 m_MajorAxis(m_dAxes(2,1),m_dAxes(2,2),m_dAxes(2,3));
  return m_MajorAxis;
}
//______________________________________________________________

TVector3 TopologyWorker::minorAxis() {
  TVector3 m_MinorAxis(m_dAxes(3,1),m_dAxes(3,2),m_dAxes(3,3));
  return m_MinorAxis;
}
//______________________________________________________________

TVector3 TopologyWorker::thrust() {
  TVector3 m_Thrust(m_dThrust[1],m_dThrust[2],m_dThrust[3]);
  return m_Thrust;
}
//______________________________________________________________

double TopologyWorker::oblateness() {
  return m_dOblateness;
}
//______________________________________________________________

// utilities(from Jetset):
double TopologyWorker::ulAngle(double x, double y)
{
  double ulangl = 0;
  double r = TMath::Sqrt(x*x + y*y);
  if ( r < 1.0E-20 ) {
    return ulangl; 
  }
  if ( TMath::Abs(x)/r < 0.8 ) {
    ulangl = sign(TMath::ACos(x/r),y);
  }
  else {
    ulangl = TMath::ASin(y/r);
    if ( x < 0. && ulangl >= 0. ) {
      ulangl = TMath::Pi() - ulangl;
    }
    else if ( x < 0. ) {
      ulangl = -TMath::Pi() - ulangl;
    }
  }
  return ulangl;
}
//______________________________________________________________

double TopologyWorker::sign(double a, double b) {
  if ( b < 0 ) {
    return -TMath::Abs(a);
  }
  else {
    return TMath::Abs(a);
  }
}
//______________________________________________________________

void TopologyWorker::ludbrb(TMatrix* mom, 
			double the, 
			double phi, 
			double bx, 
			double by,
			double bz)
{
  // Ignore "zeroth" elements in rot,pr,dp.
  // Trying to use physics-like notation.
  TMatrix rot(4,4);
  double pr[4];
  double dp[5];
  Int_t np = mom->GetNrows();
  if ( the*the + phi*phi > 1.0E-20 )
    {
      rot(1,1) = TMath::Cos(the)*TMath::Cos(phi);
      rot(1,2) = -TMath::Sin(phi);
      rot(1,3) = TMath::Sin(the)*TMath::Cos(phi);
      rot(2,1) = TMath::Cos(the)*TMath::Sin(phi);
      rot(2,2) = TMath::Cos(phi);
      rot(2,3) = TMath::Sin(the)*TMath::Sin(phi);
      rot(3,1) = -TMath::Sin(the);
      rot(3,2) = 0.0;
      rot(3,3) = TMath::Cos(the);
      for ( Int_t i = 0; i < np; i++ )
	{
	  for ( Int_t j = 1; j < 4; j++ )
	    {
	      pr[j] = (*mom)(i,j);
	      (*mom)(i,j) = 0;
	    }
	  for ( Int_t jb = 1; jb < 4; jb++)
	    {
	      for ( Int_t k = 1; k < 4; k++)
		{
		  (*mom)(i,jb) = (*mom)(i,jb) + rot(jb,k)*pr[k];
		}
	    }
	}
      double beta = TMath::Sqrt( bx*bx + by*by + bz*bz );
      if ( beta*beta > 1.0E-20 )
	{
	  if ( beta >  0.99999999 )
	    {
			 //send message: boost too large, resetting to <~1.0.
	      bx = bx*(0.99999999/beta);
	      by = by*(0.99999999/beta);
	      bz = bz*(0.99999999/beta);
	      beta =   0.99999999;
	    }
	  double gamma = 1.0/TMath::Sqrt(1.0 - beta*beta);
	  for ( Int_t i = 0; i < np; i++ )
	    {
	      for ( Int_t j = 1; j < 5; j++ )
		{
		  dp[j] = (*mom)(i,j);
		}
	      double bp = bx*dp[1] + by*dp[2] + bz*dp[3];
	      double gbp = gamma*(gamma*bp/(1.0 + gamma) + dp[4]);
	      (*mom)(i,1) = dp[1] + gbp*bx;
	      (*mom)(i,2) = dp[2] + gbp*by;
	      (*mom)(i,3) = dp[3] + gbp*bz;
	      (*mom)(i,4) = gamma*(dp[4] + bp);
	    }
	}
    }
  return;
}



// APLANARITY and SPHERICITY

void TopologyWorker::sanda() {
      float SPH=-1;
      float APL=-1;
      m_sanda_called=true;
  //=======================================================================
  // By M.Vreeswijk, (core was fortran, stolen from somewhere)
  // Purpose: to perform sphericity tensor analysis to give sphericity 
  // and aplanarity. 
  //
  // Needs: Array (standard from root-tuples): GTRACK_px, py, pz 
  //        The number of tracks in these arrays: GTRACK_ijet
  //        In addition: Array GTRACK_ijet contains a jet number 'ijet'
  //        (if you wish to change this, simply change code)
  //
  // Uses: TVector3 and TLorentzVector classes
  // 
  // Output: Sphericity SPH and Aplanarity APL
  //=======================================================================
// C...Calculate matrix to be diagonalized.
      float P[1000][6];
      double SM[4][4],SV[4][4];
      double PA,PWT,PS,SQ,SR,SP,SMAX,SGN;
      int NP;
      int J, J1, J2, I, N, JA, JB, J3, JC, JB1, JB2;
      JA=JB=JC=0;
      double RL;
      float rlu,rlu1;
      //
      // --- get the input form GTRACK arrays
      //
      N=m_np;
      NP=0;
      for (I=1;I<N+1;I++){
	   NP++; // start at one
	   P[NP][1]=m_mom(I-1,1) ;
	   P[NP][2]=m_mom(I-1,2) ;
	   P[NP][3]=m_mom(I-1,3) ;
	   P[NP][4]=m_mom(I-1,4) ;
	   P[NP][5]=0;
       }
      //
      //---
      //
       N=NP;

      for (J1=1;J1<4;J1++) {
	for (J2=J1;J2<4;J2++) {
	  SM[J1][J2]=0.;
	}
      }
      PS=0.;
      for (I=1;I<N+1;I++) { // 140
         PA=sqrt(pow(P[I][1],2)+pow(P[I][2],2)+pow(P[I][3],2)); 
         PWT=1.;
         for (J1=1;J1<4;J1++) { // 130
            for (J2=J1;J2<4;J2++) { // 120
               SM[J1][J2]=SM[J1][J2]+PWT*P[I][J1]*P[I][J2];
            }
         } // 130
         PS=PS+PWT*PA*PA;
       } //140
// C...Very low multiplicities (0 or 1) not considered.
      if(NP<2) {
        SPH=-1.;
        APL=-1.;
	return;	
      }
      for (J1=1;J1<4;J1++) { // 160
         for (J2=J1;J2<4;J2++) { // 150
            SM[J1][J2]=SM[J1][J2]/PS;
         }
      } // 160
// C...Find eigenvalues to matrix (third degree equation).
      SQ=(SM[1][1]*SM[2][2]+SM[1][1]*SM[3][3]+SM[2][2]*SM[3][3]
	  -pow(SM[1][2],2)
	  -pow(SM[1][3],2)-pow(SM[2][3],2))/3.-1./9.;
      SR=-0.5*(SQ+1./9.+SM[1][1]*pow(SM[2][3],2)+SM[2][2]*pow(SM[1][3],2)+SM[3][3]*
     pow(SM[1][2],2)-SM[1][1]*SM[2][2]*SM[3][3])+SM[1][2]*SM[1][3]*SM[2][3]+1./27.;

      SP=TMath::Cos(TMath::ACos(TMath::Max(TMath::Min(SR/TMath::Sqrt(-pow(SQ,3)),1.),-1.))/3.);

 P[N+1][4]=1./3.+TMath::Sqrt(-SQ)*TMath::Max(2.*SP,TMath::Sqrt(3.*(1.-SP*SP))-SP);
      P[N+3][4]=1./3.+TMath::Sqrt(-SQ)*TMath::Min(2.*SP,-TMath::Sqrt(3.*(1.-SP*SP))-SP);
      P[N+2][4]=1.-P[N+1][4]-P[N+3][4];
      if (P[N+2][4]> 1E-5) {
 
// C...Find first and last eigenvector by solving equation system.
      for (I=1;I<4;I=I+2) { // 240
         for (J1=1;J1<4;J1++) { // 180
            SV[J1][J1]=SM[J1][J1]-P[N+I][4];
            for (J2=J1+1;J2<4;J2++) { // 170
               SV[J1][J2]=SM[J1][J2];
               SV[J2][J1]=SM[J1][J2];
            }
          } // 180
         SMAX=0.;
         for (J1=1;J1<4;J1++) { // 200
            for (J2=1;J2<4;J2++) { // 190
              if(std::fabs(SV[J1][J2])>SMAX) { // 190
                 JA=J1;
                 JB=J2;
                 SMAX=std::fabs(SV[J1][J2]);
              }
            } // 190
          } // 200
          SMAX=0.;
	  for (J3=JA+1;J3<JA+3;J3++) { // 220
             J1=J3-3*((J3-1)/3);
             RL=SV[J1][JB]/SV[JA][JB];
             for (J2=1;J2<4;J2++) { // 210
                SV[J1][J2]=SV[J1][J2]-RL*SV[JA][J2];
                if (std::fabs(SV[J1][J2])>SMAX) { // GOTO 210
                   JC=J1;
                   SMAX=std::fabs(SV[J1][J2]);
                 }
               } // 210
            }  // 220 
            JB1=JB+1-3*(JB/3);
            JB2=JB+2-3*((JB+1)/3);
            P[N+I][JB1]=-SV[JC][JB2];
            P[N+I][JB2]=SV[JC][JB1];
            P[N+I][JB]=-(SV[JA][JB1]*P[N+I][JB1]+SV[JA][JB2]*P[N+I][JB2])/
                  SV[JA][JB];
            PA=TMath::Sqrt(pow(P[N+I][1],2)+pow(P[N+I][2],2)+pow(P[N+I][3],2));
// make a random number
	    float pa=P[N-1][I];
	    rlu=std::fabs(pa)-std::fabs(int(pa)*1.);
            rlu1=std::fabs(pa*pa)-std::fabs(int(pa*pa)*1.);
            SGN=pow((-1.),1.*int(rlu+0.5));
	    for (J=1;J<4;J++) { // 230
               P[N+I][J]=SGN*P[N+I][J]/PA;
            } // 230
      } // 240
 
// C...Middle axis orthogonal to other two. Fill other codes.      
      SGN=pow((-1.),1.*int(rlu1+0.5));
      P[N+2][1]=SGN*(P[N+1][2]*P[N+3][3]-P[N+1][3]*P[N+3][2]);
      P[N+2][2]=SGN*(P[N+1][3]*P[N+3][1]-P[N+1][1]*P[N+3][3]);
      P[N+2][3]=SGN*(P[N+1][1]*P[N+3][2]-P[N+1][2]*P[N+3][1]);
 
// C...Calculate sphericity and aplanarity. Select storing option.
      SPH=1.5*(P[N+2][4]+P[N+3][4]);
      APL=1.5*P[N+3][4];

      } // check 1

      m_sph=SPH;
      m_apl=APL;
      return;
} // end sanda




void TopologyWorker::planes_sphe(double& pnorm, double& p2, double& p3) {
  //float SPH=-1; //FIXME: commented out since gcc461 complained that this variable is set but unused
  //float APL=-1; //FIXME: commented out since gcc461 complained that this variable is set but unused
// C...Calculate matrix to be diagonalized.
      float P[1000][6];
      double SM[4][4],SV[4][4];
      double PA,PWT,PS,SQ,SR,SP,SMAX,SGN;
      int NP;
      int J, J1, J2, I, N, JA, JB, J3, JC, JB1, JB2;
      JA=JB=JC=0;
      double RL;
      float rlu,rlu1;
      //
      // --- get the input form GTRACK arrays
      //
      N=m_np;
      NP=0;
      for (I=1;I<N+1;I++){
	   NP++; // start at one
	   P[NP][1]=m_mom(I-1,1) ;
	   P[NP][2]=m_mom(I-1,2) ;
	   P[NP][3]=m_mom(I-1,3) ;
	   P[NP][4]=m_mom(I-1,4) ;
	   P[NP][5]=0;
       }
      //
      //---
      //
       N=NP;

      for (J1=1;J1<4;J1++) {
	for (J2=J1;J2<4;J2++) {
	  SM[J1][J2]=0.;
	}
      }
      PS=0.;
      for (I=1;I<N+1;I++) { // 140
         PA=sqrt(pow(P[I][1],2)+pow(P[I][2],2)+pow(P[I][3],2)); 
	 double eta,phi;
	 getetaphi(P[I][1],P[I][2],P[I][3],eta,phi);
	 PWT=exp(-std::fabs(eta));
	 PWT=1.;
         for (J1=1;J1<4;J1++) { // 130
            for (J2=J1;J2<4;J2++) { // 120
               SM[J1][J2]=SM[J1][J2]+PWT*P[I][J1]*P[I][J2];
            }
         } // 130
         PS=PS+PWT*PA*PA;
       } //140
// C...Very low multiplicities (0 or 1) not considered.
      if(NP<2) {
        //SPH=-1.; //FIXME: commented out since gcc461 complained that this variable is set but unused
        //APL=-1.; //FIXME: commented out since gcc461 complained that this variable is set but unused
	return;	
      }
      for (J1=1;J1<4;J1++) { // 160
         for (J2=J1;J2<4;J2++) { // 150
            SM[J1][J2]=SM[J1][J2]/PS;
         }
      } // 160
// C...Find eigenvalues to matrix (third degree equation).
      SQ=(SM[1][1]*SM[2][2]+SM[1][1]*SM[3][3]+SM[2][2]*SM[3][3]
	  -pow(SM[1][2],2)
	  -pow(SM[1][3],2)-pow(SM[2][3],2))/3.-1./9.;
      SR=-0.5*(SQ+1./9.+SM[1][1]*pow(SM[2][3],2)+SM[2][2]*pow(SM[1][3],2)+SM[3][3]*
     pow(SM[1][2],2)-SM[1][1]*SM[2][2]*SM[3][3])+SM[1][2]*SM[1][3]*SM[2][3]+1./27.;

      SP=TMath::Cos(TMath::ACos(TMath::Max(TMath::Min(SR/TMath::Sqrt(-pow(SQ,3)),1.),-1.))/3.);

 P[N+1][4]=1./3.+TMath::Sqrt(-SQ)*TMath::Max(2.*SP,TMath::Sqrt(3.*(1.-SP*SP))-SP);
      P[N+3][4]=1./3.+TMath::Sqrt(-SQ)*TMath::Min(2.*SP,-TMath::Sqrt(3.*(1.-SP*SP))-SP);
      P[N+2][4]=1.-P[N+1][4]-P[N+3][4];
      if (P[N+2][4]> 1E-5) {
 
// C...Find first and last eigenvector by solving equation system.
      for (I=1;I<4;I=I+2) { // 240
         for (J1=1;J1<4;J1++) { // 180
            SV[J1][J1]=SM[J1][J1]-P[N+I][4];
            for (J2=J1+1;J2<4;J2++) { // 170
               SV[J1][J2]=SM[J1][J2];
               SV[J2][J1]=SM[J1][J2];
            }
          } // 180
         SMAX=0.;
         for (J1=1;J1<4;J1++) { // 200
            for (J2=1;J2<4;J2++) { // 190
              if(std::fabs(SV[J1][J2])>SMAX) { // 190
                 JA=J1;
                 JB=J2;
                 SMAX=std::fabs(SV[J1][J2]);
              }
            } // 190
          } // 200
          SMAX=0.;
	  for (J3=JA+1;J3<JA+3;J3++) { // 220
             J1=J3-3*((J3-1)/3);
             RL=SV[J1][JB]/SV[JA][JB];
             for (J2=1;J2<4;J2++) { // 210
                SV[J1][J2]=SV[J1][J2]-RL*SV[JA][J2];
                if (std::fabs(SV[J1][J2])>SMAX) { // GOTO 210
                   JC=J1;
                   SMAX=std::fabs(SV[J1][J2]);
                 }
               } // 210
            }  // 220 
            JB1=JB+1-3*(JB/3);
            JB2=JB+2-3*((JB+1)/3);
            P[N+I][JB1]=-SV[JC][JB2];
            P[N+I][JB2]=SV[JC][JB1];
            P[N+I][JB]=-(SV[JA][JB1]*P[N+I][JB1]+SV[JA][JB2]*P[N+I][JB2])/
                  SV[JA][JB];
            PA=TMath::Sqrt(pow(P[N+I][1],2)+pow(P[N+I][2],2)+pow(P[N+I][3],2));
// make a random number
	    float pa=P[N-1][I];
	    rlu=std::fabs(pa)-std::fabs(int(pa)*1.);
            rlu1=std::fabs(pa*pa)-std::fabs(int(pa*pa)*1.);
            SGN=pow((-1.),1.*int(rlu+0.5));
	    for (J=1;J<4;J++) { // 230
               P[N+I][J]=SGN*P[N+I][J]/PA;
            } // 230
      } // 240
 
// C...Middle axis orthogonal to other two. Fill other codes.      
      SGN=pow((-1.),1.*int(rlu1+0.5));
      P[N+2][1]=SGN*(P[N+1][2]*P[N+3][3]-P[N+1][3]*P[N+3][2]);
      P[N+2][2]=SGN*(P[N+1][3]*P[N+3][1]-P[N+1][1]*P[N+3][3]);
      P[N+2][3]=SGN*(P[N+1][1]*P[N+3][2]-P[N+1][2]*P[N+3][1]);
 
// C...Calculate sphericity and aplanarity. Select storing option.
      //SPH=1.5*(P[N+2][4]+P[N+3][4]); //FIXME: commented out since gcc461 complained that this variable is set but unused
      //APL=1.5*P[N+3][4];             //FIXME: commented out since gcc461 complained that this variable is set but unused

      } // check 1

      // so assume now we have Sphericity axis, which one give the minimal Pts
      double etstot[4];
      double eltot[4];
      double sum23=0;
      double sum22=0;
      double sum33=0;
      double pina[4];
      double ax[4], ay[4], az[4];
      for (int ia=1;ia<4;ia++) {
	etstot[ia]=0;
	eltot[ia]=0;
	pina[ia]=0;
	ax[ia]=P[N+ia][1];
	ay[ia]=P[N+ia][2];
	az[ia]=P[N+ia][3];
	ax[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
	ay[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
	az[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
      }


      for (int k =0 ; k<m_np ; k++) {
	//	 double eta,phi;
	//  getetaphi(m_mom(k,1),m_mom(k,2),m_mom(k,3),eta,phi);
	double W=1.0;
	for (int ia=1;ia<4;ia++) {
	  double e=sqrt(m_mom(k,1)*m_mom(k,1) +
			m_mom(k,2)*m_mom(k,2) +
			m_mom(k,3)*m_mom(k,3));
	  double el=ax[ia]*m_mom(k,1) + ay[ia]*m_mom(k,2) + az[ia]*m_mom(k,3);
	  pina[ia]=el;
	  double ets=(e*e-el*el);
	  etstot[ia]+=ets*W;
	  eltot[ia]+=el*el;
	}
	double a2=pina[2];
	double a3=pina[3];
	//	double h=0.4;
	//a2=pina[2]*cos(h)+pina[3]*sin(h);
	//a3=pina[3]*cos(h)-pina[2]*sin(h);
	sum22+=a2*a2*W;
	sum23+=a2*a3*W;
	sum33+=a3*a3*W;
      }

      
  
	double pi=3.1415927;
	double phi=pi/2.0;
	double phip=pi/2.0;
	double a=sum23; 
	double c=-a;
	double b=sum22-sum33;
	double disc=b*b-4*a*c;
	//   cout << " disc " << disc << endl;
	if (disc>=0) {
	  double x1=(sqrt(disc)-b)/2/a;
	  double x2=(-sqrt(disc)-b)/2/a;
	  phi=atan(x1);
	  phip=atan(x2);
	  if (phi<0) phi=2.*pi+phi;
	  if (phip<0) phip=2.*pi+phip;
	}
	double p21=sum22*cos(phi)*cos(phi)+sum33*sin(phi)*sin(phi)+2*sum23*cos(phi)*sin(phi);
	double p31=sum22*sin(phi)*sin(phi)+sum33*cos(phi)*cos(phi)-2*sum23*cos(phi)*sin(phi);

	double p22=sum22*cos(phip)*cos(phip)+sum33*sin(phip)*sin(phip)+2*sum23*cos(phip)*sin(phip);
	double p32=sum22*sin(phip)*sin(phip)+sum33*cos(phip)*cos(phip)-2*sum23*cos(phip)*sin(phip);

     
	double d1=std::fabs(p31*p31 - p21*p21);
	double d2=std::fabs(p32*p32 - p22*p22);
	//cout << " eltot " << eltot[2] << " " << eltot[3] << endl;
	//cout << " phi " << phi << " " << phip << endl;
	//cout << " d " << d1 << " " << d2 << endl;
	p2=p21;
	p3=p31;
	if (d2>d1) { 
	  p2=p22;
	  p3=p32;
	}
	pnorm=sqrt(eltot[1]);
	if (p2>p3) {
	  p3=sqrt(p3);
	  p2=sqrt(p2);
	}else {
	  double p4=p3;
	  p3=sqrt(p2);
	  p2=sqrt(p4);
	}
	//cout << " sum2 sum3 " << sqrt(sum22) << " " << sqrt(sum33) << endl;
	//cout << " p2 p3 " << p2 << " " << p3 << endl;
	//double els=sqrt(eltot[2]*eltot[2]+eltot[3]*eltot[3]);
	//	cout << " ets els " << (ettot[1]) << " " << els << endl;

	//m_sph=SPH; //FIXME: shouldn't the local variables be used to reset the class member variables accordingly?
	//m_apl=APL; //FIXME: shouldn't the local variables be used to reset the class member variables accordingly?
    return;
} // end planes_sphe


void TopologyWorker::planes_sphe_wei(double& pnorm, double& p2, double& p3) {
      //float SPH=-1; //FIXME: commented out since gcc461 complained that this variable is set but unused
      //float APL=-1; //FIXME: commented out since gcc461 complained that this variable is set but unused
// C...Calculate matrix to be diagonalized.
      float P[1000][6];
      double SM[4][4],SV[4][4];
      double PA,PWT,PS,SQ,SR,SP,SMAX,SGN;
      int NP;
      int J, J1, J2, I, N, JA, JB, J3, JC, JB1, JB2;
      JA=JB=JC=0;
      double RL;
      float rlu,rlu1;
      //
      // --- get the input form GTRACK arrays
      //
      N=m_np;
      NP=0;
      for (I=1;I<N+1;I++){
	   NP++; // start at one
	   P[NP][1]=m_mom(I-1,1) ;
	   P[NP][2]=m_mom(I-1,2) ;
	   P[NP][3]=m_mom(I-1,3) ;
	   P[NP][4]=m_mom(I-1,4) ;
	   P[NP][5]=0;
       }
      //
      //---
      //
       N=NP;

      for (J1=1;J1<4;J1++) {
	for (J2=J1;J2<4;J2++) {
	  SM[J1][J2]=0.;
	}
      }
      PS=0.;
      for (I=1;I<N+1;I++) { // 140
         PA=sqrt(pow(P[I][1],2)+pow(P[I][2],2)+pow(P[I][3],2)); 
	 // double eta,phi;
	 // getetaphi(P[I][1],P[I][2],P[I][3],eta,phi);
	 //  PWT=exp(-std::fabs(eta));
	 PWT=1.;
         for (J1=1;J1<4;J1++) { // 130
            for (J2=J1;J2<4;J2++) { // 120
               SM[J1][J2]=SM[J1][J2]+PWT*P[I][J1]*P[I][J2];
            }
         } // 130
         PS=PS+PWT*PA*PA;
       } //140
// C...Very low multiplicities (0 or 1) not considered.
      if(NP<2) {
        //SPH=-1.; //FIXME: commented out since gcc461 complained that this variable is set but unused
        //APL=-1.; //FIXME: commented out since gcc461 complained that this variable is set but unused
	return;	
      }
      for (J1=1;J1<4;J1++) { // 160
         for (J2=J1;J2<4;J2++) { // 150
            SM[J1][J2]=SM[J1][J2]/PS;
         }
      } // 160
// C...Find eigenvalues to matrix (third degree equation).
      SQ=(SM[1][1]*SM[2][2]+SM[1][1]*SM[3][3]+SM[2][2]*SM[3][3]
	  -pow(SM[1][2],2)
	  -pow(SM[1][3],2)-pow(SM[2][3],2))/3.-1./9.;
      SR=-0.5*(SQ+1./9.+SM[1][1]*pow(SM[2][3],2)+SM[2][2]*pow(SM[1][3],2)+SM[3][3]*
     pow(SM[1][2],2)-SM[1][1]*SM[2][2]*SM[3][3])+SM[1][2]*SM[1][3]*SM[2][3]+1./27.;

      SP=TMath::Cos(TMath::ACos(TMath::Max(TMath::Min(SR/TMath::Sqrt(-pow(SQ,3)),1.),-1.))/3.);

 P[N+1][4]=1./3.+TMath::Sqrt(-SQ)*TMath::Max(2.*SP,TMath::Sqrt(3.*(1.-SP*SP))-SP);
      P[N+3][4]=1./3.+TMath::Sqrt(-SQ)*TMath::Min(2.*SP,-TMath::Sqrt(3.*(1.-SP*SP))-SP);
      P[N+2][4]=1.-P[N+1][4]-P[N+3][4];
      if (P[N+2][4]> 1E-5) {
 
// C...Find first and last eigenvector by solving equation system.
      for (I=1;I<4;I=I+2) { // 240
         for (J1=1;J1<4;J1++) { // 180
            SV[J1][J1]=SM[J1][J1]-P[N+I][4];
            for (J2=J1+1;J2<4;J2++) { // 170
               SV[J1][J2]=SM[J1][J2];
               SV[J2][J1]=SM[J1][J2];
            }
          } // 180
         SMAX=0.;
         for (J1=1;J1<4;J1++) { // 200
            for (J2=1;J2<4;J2++) { // 190
              if(std::fabs(SV[J1][J2])>SMAX) { // 190
                 JA=J1;
                 JB=J2;
                 SMAX=std::fabs(SV[J1][J2]);
              }
            } // 190
          } // 200
          SMAX=0.;
	  for (J3=JA+1;J3<JA+3;J3++) { // 220
             J1=J3-3*((J3-1)/3);
             RL=SV[J1][JB]/SV[JA][JB];
             for (J2=1;J2<4;J2++) { // 210
                SV[J1][J2]=SV[J1][J2]-RL*SV[JA][J2];
                if (std::fabs(SV[J1][J2])>SMAX) { // GOTO 210
                   JC=J1;
                   SMAX=std::fabs(SV[J1][J2]);
                 }
               } // 210
            }  // 220 
            JB1=JB+1-3*(JB/3);
            JB2=JB+2-3*((JB+1)/3);
            P[N+I][JB1]=-SV[JC][JB2];
            P[N+I][JB2]=SV[JC][JB1];
            P[N+I][JB]=-(SV[JA][JB1]*P[N+I][JB1]+SV[JA][JB2]*P[N+I][JB2])/
                  SV[JA][JB];
            PA=TMath::Sqrt(pow(P[N+I][1],2)+pow(P[N+I][2],2)+pow(P[N+I][3],2));
// make a random number
	    float pa=P[N-1][I];
	    rlu=std::fabs(pa)-std::fabs(int(pa)*1.);
            rlu1=std::fabs(pa*pa)-std::fabs(int(pa*pa)*1.);
            SGN=pow((-1.),1.*int(rlu+0.5));
	    for (J=1;J<4;J++) { // 230
               P[N+I][J]=SGN*P[N+I][J]/PA;
            } // 230
      } // 240
 
// C...Middle axis orthogonal to other two. Fill other codes.      
      SGN=pow((-1.),1.*int(rlu1+0.5));
      P[N+2][1]=SGN*(P[N+1][2]*P[N+3][3]-P[N+1][3]*P[N+3][2]);
      P[N+2][2]=SGN*(P[N+1][3]*P[N+3][1]-P[N+1][1]*P[N+3][3]);
      P[N+2][3]=SGN*(P[N+1][1]*P[N+3][2]-P[N+1][2]*P[N+3][1]);
 
// C...Calculate sphericity and aplanarity. Select storing option.
      //SPH=1.5*(P[N+2][4]+P[N+3][4]); //FIXME: commented out since gcc461 complained that this variable is set but unused
      //APL=1.5*P[N+3][4];             //FIXME: commented out since gcc461 complained that this variable is set but unused

      } // check 1

      // so assume now we have Sphericity axis, which one give the minimal Pts
      double etstot[4];
      double eltot[4];
      double sum23=0;
      double sum22=0;
      double sum33=0;
      double pina[4];
      double ax[4], ay[4], az[4];
      for (int ia=1;ia<4;ia++) {
	etstot[ia]=0;
	eltot[ia]=0;
	pina[ia]=0;
	ax[ia]=P[N+ia][1];
	ay[ia]=P[N+ia][2];
	az[ia]=P[N+ia][3];
	ax[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
	ay[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
	az[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
      }

      for (int k =0 ; k<m_np ; k++) {
	 double eta,phi;
	 getetaphi(m_mom(k,1),m_mom(k,2),m_mom(k,3),eta,phi);
	 double W=exp(-std::fabs(eta*1.0));
	for (int ia=1;ia<4;ia++) {
	  double e=sqrt(m_mom(k,1)*m_mom(k,1) +
			m_mom(k,2)*m_mom(k,2) +
			m_mom(k,3)*m_mom(k,3));
	  double el=ax[ia]*m_mom(k,1) + ay[ia]*m_mom(k,2) + az[ia]*m_mom(k,3);
	  pina[ia]=el;
	  double ets=(e*e-el*el);
	  etstot[ia]+=ets*W;
	  eltot[ia]+=el*el*W;
	}
	double a2=pina[2];
	double a3=pina[3];
	//	double h=0.4;
	//a2=pina[2]*cos(h)+pina[3]*sin(h);
	//a3=pina[3]*cos(h)-pina[2]*sin(h);
	sum22+=a2*a2*W;
	sum23+=a2*a3*W;
	sum33+=a3*a3*W;
      }
      
  
	double pi=3.1415927;
	double phi=pi/2.0;
	double phip=pi/2.0;
	double a=sum23; 
	double c=-a;
	double b=sum22-sum33;
	double disc=b*b-4*a*c;
	//   cout << " disc " << disc << endl;
	if (disc>=0) {
	  double x1=(sqrt(disc)-b)/2/a;
	  double x2=(-sqrt(disc)-b)/2/a;
	  phi=atan(x1);
	  phip=atan(x2);
	  if (phi<0) phi=2.*pi+phi;
	  if (phip<0) phip=2.*pi+phip;
	}
	double p21=sum22*cos(phi)*cos(phi)+sum33*sin(phi)*sin(phi)+2*sum23*cos(phi)*sin(phi);
	double p31=sum22*sin(phi)*sin(phi)+sum33*cos(phi)*cos(phi)-2*sum23*cos(phi)*sin(phi);

	double p22=sum22*cos(phip)*cos(phip)+sum33*sin(phip)*sin(phip)+2*sum23*cos(phip)*sin(phip);
	double p32=sum22*sin(phip)*sin(phip)+sum33*cos(phip)*cos(phip)-2*sum23*cos(phip)*sin(phip);

     
	double d1=std::fabs(p31*p31 - p21*p21);
	double d2=std::fabs(p32*p32 - p22*p22);
	//cout << " eltot " << eltot[2] << " " << eltot[3] << endl;
	//cout << " phi " << phi << " " << phip << endl;
	//cout << " d " << d1 << " " << d2 << endl;
	p2=p21;
	p3=p31;
	if (d2>d1) { 
	  p2=p22;
	  p3=p32;
	}
	pnorm=sqrt(eltot[1]);
	if (p2>p3) {
	  p3=sqrt(p3);
	  p2=sqrt(p2);
	}else {
	  double p4=p3;
	  p3=sqrt(p2);
	  p2=sqrt(p4);
	}
	//cout << " sum2 sum3 " << sqrt(sum22) << " " << sqrt(sum33) << endl;
	//cout << " p2 p3 " << p2 << " " << p3 << endl;
	//double els=sqrt(eltot[2]*eltot[2]+eltot[3]*eltot[3]);
	//	cout << " ets els " << (ettot[1]) << " " << els << endl;

	//m_sph=SPH; //FIXME: shouldn't the local variables be used to reset the class member variables accordingly?
	//m_apl=APL; //FIXME: shouldn't the local variables be used to reset the class member variables accordingly?
    return;
} // end planes_sphe



void TopologyWorker::planes_thrust(double& pnorm, double& p2, double& p3) {
	TVector3 thrustaxis=thrustAxis();
	TVector3 majoraxis=majorAxis();
	TVector3 minoraxis=minorAxis();
      // so assume now we have Sphericity axis, which one give the minimal Pts
      double etstot[4];
      double eltot[4];
      double sum23=0;
      double sum22=0;
      double sum33=0;
      double pina[4];
      double ax[4], ay[4], az[4];
      ax[1]=thrustaxis(0); ay[1]=thrustaxis(1); az[1]=thrustaxis(2);
      ax[2]=minoraxis(0); ay[2]=minoraxis(1); az[2]=minoraxis(2);
      ax[3]=majoraxis(0); ay[3]=majoraxis(1); az[3]=majoraxis(2);
      for (int ia=1;ia<4;ia++) {
	etstot[ia]=0;
	eltot[ia]=0;
	pina[ia]=0;
	ax[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
	ay[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
	az[ia]/=sqrt(ax[ia]*ax[ia]+ay[ia]*ay[ia]+az[ia]*az[ia]);
      }

      for (int k =0 ; k<m_np ; k++) {
	for (int ia=1;ia<4;ia++) {
	  double e=sqrt(m_mom(k,1)*m_mom(k,1) +
			m_mom(k,2)*m_mom(k,2) +
			m_mom(k,3)*m_mom(k,3));
	  double el=ax[ia]*m_mom(k,1) + ay[ia]*m_mom(k,2) + az[ia]*m_mom(k,3);
	  pina[ia]=el;
	  double ets=(e*e-el*el);
	  etstot[ia]+=ets;
	  eltot[ia]+=std::fabs(el);
	}
	double a2=pina[2];
	double a3=pina[3];
	//	double h=0.4;
	//a2=pina[2]*cos(h)+pina[3]*sin(h);
	//a3=pina[3]*cos(h)-pina[2]*sin(h);
	sum22+=a2*a2;
	sum23+=a2*a3;
	sum33+=a3*a3;
      }
      
  
	double pi=3.1415927;
	double phi=pi/2.0;
	double phip=pi/2.0;
	double a=sum23; 
	double c=-a;
	double b=sum22-sum33;
	double disc=b*b-4*a*c;
	//   cout << " disc " << disc << endl;
	if (disc>=0) {
	  double x1=(sqrt(disc)-b)/2/a;
	  double x2=(-sqrt(disc)-b)/2/a;
	  phi=atan(x1);
	  phip=atan(x2);
	  if (phi<0) phi=2.*pi+phi;
	  if (phip<0) phip=2.*pi+phip;
	}
	double p21=sum22*cos(phi)*cos(phi)+sum33*sin(phi)*sin(phi)+2*sum23*cos(phi)*sin(phi);
	double p31=sum22*sin(phi)*sin(phi)+sum33*cos(phi)*cos(phi)-2*sum23*cos(phi)*sin(phi);

	double p22=sum22*cos(phip)*cos(phip)+sum33*sin(phip)*sin(phip)+2*sum23*cos(phip)*sin(phip);
	double p32=sum22*sin(phip)*sin(phip)+sum33*cos(phip)*cos(phip)-2*sum23*cos(phip)*sin(phip);

     
	double d1=std::fabs(p31*p31 - p21*p21);
	double d2=std::fabs(p32*p32 - p22*p22);
	//cout << " eltot " << eltot[2] << " " << eltot[3] << endl;
	//cout << " phi " << phi << " " << phip << endl;
	//cout << " d " << d1 << " " << d2 << endl;
	p2=p21;
	p3=p31;
	if (d2>d1) { 
	  p2=p22;
	  p3=p32;
	}
	pnorm=sqrt(etstot[1]);
	if (p2>p3) {
	  p3=sqrt(p3);
	  p2=sqrt(p2);
	}else {
	  double p4=p3;
	  p3=sqrt(p2);
	  p2=sqrt(p4);
	}
	//cout << " sum2 sum3 " << sqrt(sum22) << " " << sqrt(sum33) << endl;
	//cout << " p2 p3 " << p2 << " " << p3 << endl;
	//double els=sqrt(eltot[2]*eltot[2]+eltot[3]*eltot[3]);
	//	cout << " ets els " << (ettot[1]) << " " << els << endl;
    return;
} // end planes_thru


void TopologyWorker::fowo() {
// 20020830 changed: from p/E to Et/Ettot and include H50 and H60
  m_fowo_called=true;
    // include fox wolframs
    float H10=-1;
    float H20=-1;
    float H30=-1;
    float H40=-1;
    float H50=-1;
    float H60=-1;
    if (1==1) {
      float P[1000][6],H0,HD,CTHE;
      int N,NP,I,J,I1,I2;      
      H0=HD=0.;
      N=m_np;
      NP=0;
      for (I=1;I<N+1;I++){
	   NP++; // start at one
	   P[NP][1]=m_mom(I-1,1) ;
	   P[NP][2]=m_mom(I-1,2) ;
	   P[NP][3]=m_mom(I-1,3) ;
	   P[NP][4]=m_mom(I-1,4) ;
	   P[NP][5]=m_mom(I-1,5) ;
       }

       N=NP;
       NP=0;

       for (I=1;I<N+1;I++) {
	 NP=NP+1;
	 for (J=1;J<5;J++) {
	   P[N+NP][J]=P[I][J];
	 }
// p/E
 	 P[N+NP][4]=sqrt(pow(P[I][1],2)+pow(P[I][2],2)+pow(P[I][3],2));
// Et/Ettot
	 P[N+NP][5]=sqrt(pow(P[I][1],2)+pow(P[I][2],2));
	 H0=H0+P[N+NP][5];
	 HD=HD+pow(P[N+NP][5],2);
       }
       H0=H0*H0;
       
       
       
       // Very low multiplicities (0 or 1) not considered.
       if (NP<2) {
	 H10=-1.;
	 H20=-1.;
	 H30=-1.;
	 H40=-1.;
	 H50=-1.;
	 H60=-1.;
	 return;
       }
 
       // Calculate H1 - H4.
       H10=0.;
       H20=0.;
       H30=0.;
       H40=0.;
       H50=0.;
       H60=0.;
       for (I1=N+1;I1<N+NP+1;I1++) { //130
	 for (I2=I1+1;I2<N+NP+1;I2++) { // 120
	   CTHE=(P[I1][1]*P[I2][1]+P[I1][2]*P[I2][2]+P[I1][3]*P[I2][3])/
	     (P[I1][4]*P[I2][4]);
	   H10=H10+P[I1][5]*P[I2][5]*CTHE;
	   double C2=(1.5*CTHE*CTHE-0.5);
	   H20=H20+P[I1][5]*P[I2][5]*C2;
	   double C3=(2.5*CTHE*CTHE*CTHE-1.5*CTHE);
	   H30=H30+P[I1][5]*P[I2][5]*C3;
           // use recurrence
           double C4=(7*CTHE*C3 - 3*C2)/4.;
           double C5=(9*CTHE*C4 - 4*C3)/5.;
           double C6=(11*CTHE*C5 - 5*C4)/6.;
//	   H40=H40+P[I1][5]*P[I2][5]*(4.375*pow(CTHE,4)-3.75*CTHE*CTHE+0.375);
//	   H50=H50+P[I1][5]*P[I2][5]*
//         (63./8.*pow(CTHE,5)-70./8.*CTHE*CTHE*CTHE+15./8.*CTHE);
//	   H60=H60+P[I1][5]*P[I2][5]*
//         (231/16.*pow(CTHE,6)-315./16.*CTHE*CTHE*CTHE*CTHE+105./16.*CTHE*CTHE-5./16.);
	   H40=H40+P[I1][5]*P[I2][5]*C4;
	   H50=H50+P[I1][5]*P[I2][5]*C5;
	   H60=H60+P[I1][5]*P[I2][5]*C6;
	 } // 120
       } // 130
 
       H10=(HD+2.*H10)/H0;
       H20=(HD+2.*H20)/H0;
       H30=(HD+2.*H30)/H0;
       H40=(HD+2.*H40)/H0;
       H50=(HD+2.*H50)/H0;
       H60=(HD+2.*H60)/H0;
       m_h10=H10;
       m_h20=H20;
       m_h30=H30;
       m_h40=H40;
       m_h50=H50;
       m_h60=H60;
    }

}

double TopologyWorker::get_h10() {
  if (!m_fowo_called) fowo();
  return m_h10;
}
double TopologyWorker::get_h20() {
  if (!m_fowo_called) fowo();
  return m_h20;
}
double TopologyWorker::get_h30() {
  if (!m_fowo_called) fowo();
  return m_h30;
}
double TopologyWorker::get_h40() {
  if (!m_fowo_called) fowo();
  return m_h40;
}

double TopologyWorker::get_h50() {
  if (!m_fowo_called) fowo();
  return m_h50;
}
double TopologyWorker::get_h60() {
  if (!m_fowo_called) fowo();
  return m_h60;
}


double TopologyWorker::get_sphericity() {
  if (!m_sanda_called) sanda();
  return m_sph;
}
double TopologyWorker::get_aplanarity() {
  if (!m_sanda_called) sanda();
  return m_apl;
}

void TopologyWorker::getetaphi(double px, double py, double pz, double& eta, double& phi) {

  double pi=3.1415927;

  double p=sqrt(px*px+py*py+pz*pz);
  // get eta and phi
  double th=pi/2.;
  if (p!=0) {
    th=acos(pz/p); // Theta
  }
  float thg=th;
  if (th<=0) {
    thg = pi + th;
  }
  eta=-9.;
  if (tan( thg/2.0 )>0.000001) {
    eta = -log( tan( thg/2.0 ) );    
  }
  phi = atan2(py,px);
  if(phi<=0) phi += 2.0*pi;
  return;
}



void TopologyWorker::sumangles(float& sdeta, float& sdr) {
  double eta1,eta2,phi1,phi2,deta,dphi,dr;
  m_sumangles_called=true;
  sdeta=0;
  sdr=0;
  for (int k=0;k<m_np;k++){
    for (int kp=k;kp<m_np;kp++){
      getetaphi(m_mom(k,1) , m_mom(k,2), m_mom(k,3), eta1,phi1);
      getetaphi(m_mom(kp,1) , m_mom(kp,2), m_mom(kp,3), eta2,phi2);
      dphi=std::fabs(phi1-phi2);
      if (dphi>3.1415) dphi=2*3.1415-dphi;
      deta=std::fabs(eta1-eta2);
      dr=sqrt(dphi*dphi+deta*deta);
      sdeta+=deta;
      sdr+=dr;
    }
  }
  return;
}

//______________________________________________________________

Int_t TopologyWorker::iPow(Int_t man, Int_t exp)
{
  Int_t ans = 1;
  for( Int_t k = 0; k < exp; k++)
    {
      ans = ans*man;
    }
  return ans;
}

// added by Freya:

void TopologyWorker::CalcWmul(){

  Int_t njets = m_np;
  double result=0;
  for(Int_t ijet=0; ijet<njets-1; ijet++){
    double emin=55;
    double emax=55;
    if(CalcPt(ijet)<55)
      emax=CalcPt(ijet);
    if(CalcPt(ijet+1)<55)
      emin=CalcPt(ijet+1);
    result+=0.5 * (emax*emax-emin*emin)*(ijet+1);
  }
  double elo=15;
  if(CalcPt(njets-1)>elo){
    elo=CalcPt(njets-1);
  }

  result+=0.5 * (elo*elo-(15*15))*(njets);
  result/=((55*55)-100)/2.0;
  m_njetsweighed=result;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

void TopologyWorker::CalcSqrts(){
  TLorentzVector event(0,0,0,0);
  TLorentzVector worker(0,0,0,0);

  for(int i=0; i< m_np; i++){
    double energy=m_mom(i,4);
    if(m_mom(i,4)<0.00001)
      energy=sqrt(pow(m_mom(i,1),2)+pow(m_mom(i,2),2)+pow(m_mom(i,3),2));
    // assume massless particle if only TVector3s are provided...
    worker.SetXYZT(m_mom(i,1),m_mom(i,2),m_mom(i,3),energy);
    event+=worker;
  }
  m_sqrts=event.M();
}

//++++++++++++++++++++++++++++++++++++++
void TopologyWorker::CalcHTstuff(){
  m_ht=0;
  m_ht3=0;
  m_et56=0;
  m_et0=0;
  double ptlead=0;
  double h=0;
  for(int i=0; i< m_np; i++){
    //cout << i << "/" << m_np << ":" << CalcPt(i) <<  endl;
    m_ht+=CalcPt(i);
    h+=m_mom(i,4);
    if(i>1)
      m_ht3+=CalcPt(i);
    if(i==5)
      m_et56=sqrt(CalcPt(i)*CalcPt(i-1));
  }
  
  for(int j=0; j< m_np2; j++){
    //cout << j << "/" << m_np2 << ":" << CalcPt2(j) <<  endl;
    if(ptlead<CalcPt2(j))
      ptlead=CalcPt2(j);

  }
  if(m_ht>0.0001){
    m_et0=ptlead/m_ht;
    //cout << "calculating ETO" << m_et0 << "=" << ptlead << endl;
  }
  if(h>0.00001)
    m_centrality=m_ht/h;
}

#endif




