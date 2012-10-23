#include <math.h>

#include "SimDataFormats/SLHC/interface/DTMatch.h"
#include "SimDataFormats/SLHC/interface/DTMatchPt.h"

/**************************************************************************
  Only stubs and (primary) vertex
***************************************************************************/

void DTMatchPt::radius_of_curvature(const edm::ParameterSet& pSet,
			     float const x[], float const y[])
{

  float min_invRb = pSet.getUntrackedParameter<double>("min_invRb");
  float max_invRb = pSet.getUntrackedParameter<double>("max_invRb");

  /*
    It would be nice, if not mandatory, to give a sign!!!!!!!!!!!!!!!!!!
  */

  double L1sq = (x[0] - x[1])*(x[0] - x[1]) + (y[0] - y[1])*(y[0] - y[1]);
  double L1 = sqrt(L1sq);
  double L2sq = (x[1] - x[2])*(x[1] - x[2]) + (y[1] - y[2])*(y[1] - y[2]);
  double L2 = sqrt(L2sq);
  double L1DotL2 = (x[0] - x[1])*(x[1] - x[2]) + (y[0] - y[1])*(y[1] - y[2]);
  double cosine_of_bending = L1DotL2/(L1*L2);
  double delta = acos( cosine_of_bending );
  //  float _bending = 
  //    static_cast<float>( (delta <= TMath::Pi())?delta:(TMath::Pi()-delta) );
  _invRb = 2. * static_cast<float>(delta/(L1+L2));
  
  if( _invRb < min_invRb || _invRb > max_invRb ) {
    _Rb = _invRb = NAN; 
    return; 
  }
  _Rb = static_cast<float>(0.5*(L1+L2)/delta);
} 
 




void DTMatchPt::radius_of_curvature(const edm::ParameterSet& pSet,
					const GlobalVector P[])
{

  float min_invRb = pSet.getUntrackedParameter<double>("min_invRb");
  float max_invRb = pSet.getUntrackedParameter<double>("max_invRb");

  /*
    It would be nice, if not mandatory, to give a sign!!!!!!!!!!!!!!!!!!
  */

  double r2   = P[2].perp();
  double phi2 = phiCMS(P[2]); 
  double r1   = P[1].perp();
  double phi1 = phiCMS(P[1]); 
  
  if(P[0].perp() == 0.) {
    // here we are using the principal vertex constraint at its nominal coordinates! 
    _Rb = 0.5*fabs(r2-r1)/fabs(phi2-phi1); //+ 0.25*r2*r1*fabs(phi2-phi1)/fabs(r2-r1);
    if( !isnan(_Rb) )
      _invRb = 1/_Rb;
    if( _invRb < min_invRb || _invRb > max_invRb ) {
      _Rb = _invRb = NAN; 
    }
    return;
  }
  // the more general case follows:
  double r0 = P[0].perp();
  double phi0 = phiCMS(P[0]);  
  double L01sq = r0*r0 + r1*r1 - 2*r0*r1*cos(phi0-phi1);
  double L01 = sqrt(L01sq);
  double L12sq = r1*r1 + r2*r2 - 2*r1*r2*cos(phi1-phi2);
  double L12 = sqrt(L12sq);
  double L01DotL12 = 
    r0*r1*cos(phi0-phi1) - r0*r2*cos(phi0-phi2) + r1*r2*cos(phi1-phi2) - r1*r1 ;
  double cosine_of_bending = L01DotL12/(L01*L12);
  double delta = acos( cosine_of_bending );
  //  float _bending = 
  //    static_cast<float>( (delta <= TMath::Pi())?delta:(TMath::Pi()-delta) );
  _invRb = 2. * static_cast<float>(delta/(L01+L12));
  if( _invRb < min_invRb || _invRb > max_invRb ) {
    _Rb = _invRb = NAN;
    return; 
  }
  _Rb = static_cast<float>(0.5*(L01+L12)/delta);
  return;
}





void DTMatchPt::radius_of_curvature(const edm::ParameterSet& pSet,
					const float vstub_rho, const float vstub_phi,
					const GlobalVector P[])
{

  float min_invRb = pSet.getUntrackedParameter<double>("min_invRb");
  float max_invRb = pSet.getUntrackedParameter<double>("max_invRb");

  /*
    It would be nice, if not mandatory, to give a sign!!!!!!!!!!!!!!!!!!
  */

  double r2   = vstub_rho;
  double phi2 = vstub_phi;
  double r1   = P[1].perp();
  double phi1 = phiCMS(P[1]); 

  if(P[0].perp() == 0.) { 
    // vertex constraint
    _Rb = 0.5*fabs(r2-r1)/fabs(phi2-phi1); //+ 0.25*r2*r1*fabs(phi2-phi1)/fabs(r2-r1);
    if( !isnan(_Rb) ) 
      _invRb = 1/_Rb;
    else 
    if( _invRb < min_invRb || _invRb > max_invRb ) {
      _Rb = _invRb = NAN; 
    }
    return;
  }
  // the more general case:
  double r0 = P[0].perp();
  double phi0 = phiCMS(P[0]);  
  double L01sq = r0*r0 + r1*r1 - 2*r0*r1*cos(phi0-phi1);
  double L01 = sqrt(L01sq);
  double L12sq = r1*r1 + r2*r2 - 2*r1*r2*cos(phi1-phi2);
  double L12 = sqrt(L12sq);
  double L01DotL12 = 
    r0*r1*cos(phi0-phi1) - r0*r2*cos(phi0-phi2) + r1*r2*cos(phi1-phi2) - r1*r1 ;
  double cosine_of_bending = L01DotL12/(L01*L12);
  double delta = acos( cosine_of_bending );
  //  float _bending = 
  //    static_cast<float>( (delta <= TMath::Pi())?delta:(TMath::Pi()-delta) );
  _invRb = 2. * static_cast<float>(delta/(L01+L12));
  if( _invRb < min_invRb || _invRb > max_invRb ) {
    _Rb = _invRb = NAN; 
    return; 
  }
  _Rb = static_cast<float>(0.5*(L01+L12)/delta);
  return;
}





void DTMatchPt::computePt(const edm::ParameterSet& pSet,
			      float const X[], float const Y[], float const corr) {
  // using DT muons or virtual stubs
  radius_of_curvature(pSet, X, Y);
  if( isnan( _Rb ) )
    return;
  _Pt = 0.003 * _Rb * 3.8;
  _Pt += corr * _Pt;
  _invPt = 1./_Pt;
  return;
} 





void DTMatchPt::computePt(const edm::ParameterSet& pSet,
			      const GlobalVector P[], float const corr) {
  // "all stubs" approach
  radius_of_curvature(pSet, P);
  if( isnan( _Rb ) )
    return;
  _Pt = 0.003 * _Rb * 3.8;
  _Pt += corr * _Pt;
  _invPt = 1./_Pt;
  return;
} 





void  DTMatchPt::computePt(const edm::ParameterSet& pSet, 
			       const float vstub_rho, const float vstub_phi, 
			       const GlobalVector P[], 
			       float const corr)
{
  radius_of_curvature(pSet, vstub_rho, vstub_phi, P);
  if( isnan( _Rb ) )
    return;
  _Pt = 0.003 * _Rb * 3.8;
  _Pt += corr * _Pt;
  _invPt = 1./_Pt;
  return;
}





void DTMatchPt::computePt_etc(const edm::ParameterSet& pSet,
				  const GlobalVector P[], float const corr) 
{
  float min_invRb = pSet.getUntrackedParameter<double>("min_invRb");
  float max_invRb = pSet.getUntrackedParameter<double>("max_invRb");

  double r0   = P[0].perp();
  double phi0 = phiCMS(P[0]);
  double r1   = P[1].perp();
  double phi1 = phiCMS(P[1]); 
  double r2   = P[2].perp();
  double phi2 = phiCMS(P[2]);

  double Delta = r0*r0*(r1-r2) + r1*r1*(r2-r0) + r2*r2*(r0-r1);
  if( Delta == 0. || Delta == NAN )
    return;
  double invDelta = 1/Delta;
  _invRb = 
    -2*(phi0*r0*(r1-r2) + phi1*r1*(r2-r0) + phi2*r2*(r0-r1))*invDelta;
  short charge = (_invRb > 0.) - (_invRb < 0.);
  // Mu particle (-1 charge) has PDG Id 13 !!!
  //  if(charge > 0)
  //    cout << "\t\t\twrong charge " << charge << endl;
  _invRb = fabs(_invRb);
  if( fabs(_invRb) < min_invRb || fabs(_invRb) > max_invRb ) {
    _Rb = _invRb = _alpha0 = _d = NAN; 
    return; 
  }
  _Rb = static_cast<float>(1/_invRb);
  _Pt = 0.003 * _Rb * 3.8;
  _Pt += corr * _Pt;
  _invPt = 1./_Pt;
  _alpha0 = charge*invDelta*
    (phi0*r0*(r1*r1-r2*r2) + phi1*r1*(r2*r2-r0*r0) + phi2*r2*(r0*r0-r1*r1));
  _d = charge*r0*r1*r2*(phi0*(r1-r2) + phi1*(r2-r0) + phi2*(r0-r1))*invDelta;
  /*
  cout << "_Pt = " << _Pt << endl;
  cout << "_alpha0 = " << _alpha0 << endl;
  cout << "_d = " << _d << endl;
  */
  return;
}




/********************************************************************************
 ********************************************************************************
 *
 *           The constructors
 *
 ********************************************************************************
 ********************************************************************************/


DTMatchPt::DTMatchPt(string const s, int station,
		     const edm::ParameterSet& pSet,
		     const vector<TrackerTracklet*> TkTracklets):
  _label(s)  
{
  _Rb = _invRb = NAN;
  _Pt = _invPt = NAN;
  _alpha0 = _d = NAN;
  size_t sl = 0;
  if(_label == string("TrackletSL1")) 
    sl = 1;
  else if (_label == string("TrackletSL4"))
    sl = 2; 
  //  cout << TkTracklets[sl] << endl;
  if( TkTracklets[sl] == 0 )
    return;   
  if( TkTracklets[sl]->valid() && TkTracklets[sl]->PTFlag() )
    _Pt = TkTracklets[sl]->pt();
}




DTMatchPt::DTMatchPt(string const s, int station,
		     const edm::ParameterSet& pSet, 
		     const GlobalVector stub_position[],
		     bool const flagMatch[]):
  _label(s)  
{

  GlobalVector P[3];

  _Rb = _invRb = NAN;
  _Pt = _invPt = NAN;
  _alpha0 = _d = NAN;

  if( _label == string("Stubs_9_3_0") 
      && flagMatch[5] && flagMatch[3] && flagMatch[0] ) {
    P[2] = stub_position[5];
    P[1] = stub_position[3];
    P[0] = stub_position[0];
    computePt(pSet, P);
    return;
  } 
  if( _label == string("Stubs_9_1_0") 
      && flagMatch[5] && flagMatch[1] && flagMatch[0] ) {
    P[2] = stub_position[5];
    P[1] = stub_position[1];
    P[0] = stub_position[0];
    computePt(pSet, P);
    return;
  }
  if( _label == string("Stubs_3_2_0") 
      && flagMatch[3] && flagMatch[2] && flagMatch[0] ) {
    P[2] = stub_position[3];
    P[1] = stub_position[2];
    P[0] = stub_position[0];
    computePt(pSet, P);
    return;
  }
  if( _label == string("Stubs_3_1_0") 
      && flagMatch[3] && flagMatch[1] && flagMatch[0] ) {
    P[2] = stub_position[3];
    P[1] = stub_position[1];
    P[0] = stub_position[0];
    computePt(pSet, P);
    return;
  }

  //******************
  // Next use vertex:
  //******************

  P[0] = GlobalVector();

  if( _label == string("Stubs_9_3_V") && flagMatch[5] && flagMatch[3] ) { 
    P[2] = stub_position[5];
    P[1] = stub_position[3];
    computePt(pSet, P);
    return;
  }
  if( _label == string("Stubs_9_1_V") && flagMatch[5] && flagMatch[1] ) { 
    P[2] = stub_position[5];
    P[1] = stub_position[1];
    computePt(pSet, P);
    return;
  }
  if( _label == string("Stubs_9_0_V") && flagMatch[5] && flagMatch[0] ) { 
    P[2] = stub_position[5];
    P[1] = stub_position[0];
    computePt(pSet, P);
    return;
  }
 if( _label == string("Stubs_3_1_V") && flagMatch[3] && flagMatch[1] ) { 
    P[2] = stub_position[3];
    P[1] = stub_position[1];
    computePt(pSet, P);
    return;
  }
  if( _label == string("Stubs_3_0_V") && flagMatch[3] && flagMatch[0] ) { 
    P[2] = stub_position[3];
    P[1] = stub_position[0];
    computePt(pSet, P);
    return;
  }
}




/**************************************************************************
   Linearized algorithm getting _alpha0 and _d
***************************************************************************/
DTMatchPt::DTMatchPt(string const s, int station,
		     const GlobalVector stub_position[],
		     bool const flagMatch[],
		     const edm::ParameterSet& pSet):
  _label(s)  
{

  GlobalVector P[3];

  _Rb = _invRb = NAN;
  _Pt = _invPt = NAN;
  _alpha0 = _d = NAN;

  if( _label == string("LinStubs_9_3_0") 
      && flagMatch[5] && flagMatch[3] && flagMatch[0] ) {
    P[2] = stub_position[5];
    P[1] = stub_position[3];
    P[0] = stub_position[0];
    computePt_etc(pSet, P);
    return;
  } 
  if( _label == string("LinStubs_9_1_0") 
      && flagMatch[5] && flagMatch[1] && flagMatch[0] ) {
    P[2] = stub_position[5];
    P[1] = stub_position[1];
    P[0] = stub_position[0];
    computePt_etc(pSet, P);
    return;
  }
  if( _label == string("LinStubs_3_2_0") 
      && flagMatch[3] && flagMatch[2] && flagMatch[0] ) {
    P[2] = stub_position[3];
    P[1] = stub_position[2];
    P[0] = stub_position[0];
    computePt_etc(pSet, P);
    return;
  }
  if( _label == string("LinStubs_3_1_0") 
      && flagMatch[3] && flagMatch[1] && flagMatch[0] ) {
    P[2] = stub_position[3];
    P[1] = stub_position[1];
    P[0] = stub_position[0];
    computePt_etc(pSet, P);
    return;
  }
}





/**************************************************************************
   Using DT muons
***************************************************************************/
DTMatchPt::DTMatchPt(string const s, int station,
		     const edm::ParameterSet& pSet,
		     float const DTmu_x, float const DTmu_y,
		     const vector<TrackerTracklet*> TkTracklets,
		     float const stub_x[], float const stub_y[],
		     bool const flagMatch[]):
  _label(s)  
{

  float corr = 0.0;
  float X[3], Y[3]; 

  _Rb = _invRb = NAN;
  _Pt = _invPt = NAN;
  _alpha0 = _d = NAN;

  if(isnan(DTmu_x) || isnan(DTmu_y)) 
    return;

  X[0] = DTmu_x;
  Y[0] = DTmu_y;

  // Mu-tracklet-stub *******************
  size_t sl = 0;
  size_t stu = 0;
  bool OK = false;

  if(_label == string("Mu_SL4_0")) {
    sl = 2;
    OK = true;
  }
  else if(_label == string("Mu_SL4_3")) {
    sl = 2;
    stu = 3;
    OK = true;
  }
  else if(_label == string("Mu_SL1_0")) {
    sl = 1;
    OK = true;
  }
  else if(_label == string("Mu_SL1_9")) {
    sl = 1;
    stu = 5;
    OK = true;
  }
  else if(_label == string("Mu_SL0_3")) {
    stu = 3;
    OK = true;
  }
  else if(_label == string("Mu_SL0_9")) {
    stu = 5;
    OK = true;
  }
  if( OK ) {
    if( TkTracklets[sl] == 0 )
      return;  
    if( TkTracklets[sl]->valid() ) {
      X[1] = TkTracklets[sl]->rho()*cos(TkTracklets[sl]->phi()); 
      Y[1] = TkTracklets[sl]->rho()*sin(TkTracklets[sl]->phi()); 
      X[2] = stub_x[stu];
      Y[2] = stub_y[stu];
      computePt(pSet, X, Y, corr);
    }
    return;
  }
  // Mu-tracklet-tracklet ****************
  if(_label == string("Mu_SL0_SL1")) {
    stu = 1;
    OK = true;
  }
  else if(_label == string("Mu_SL0_SL4")) {
    stu = 2;
    OK = true;
  }
  else if(_label == string("Mu_SL1_SL4")) {
    sl = 1;
    stu = 2;
    OK = true;
  }
  if( OK ) {
    if( TkTracklets[sl] == 0 || TkTracklets[stu] == 0 )
      return;  
    if( TkTracklets[sl]->valid() && TkTracklets[stu]->valid() ) {
      X[2] = TkTracklets[sl]->rho()*cos(TkTracklets[sl]->phi()); 
      Y[2] = TkTracklets[sl]->rho()*sin(TkTracklets[sl]->phi()); 
      X[1] = TkTracklets[stu]->rho()*cos(TkTracklets[stu]->phi()); 
      Y[1] = TkTracklets[stu]->rho()*sin(TkTracklets[stu]->phi()); 
      computePt(pSet, X, Y, corr);
    }
    return;
  }
  // Mu-tracklet-vertex *****************
  X[2] = 0.;
  Y[2] = 0.;
  if(_label == string("Mu_SL4_V"))
    sl = 2;
  else if(_label == string("Mu_SL1_V"))
    sl = 1;
  if( TkTracklets[sl] == 0 )
    return;  
  if( TkTracklets[sl]->valid() ) {
    X[1]  = TkTracklets[sl]->rho()*cos(TkTracklets[sl]->phi()); 
    Y[1]  = TkTracklets[sl]->rho()*sin(TkTracklets[sl]->phi()); 
    computePt(pSet, X, Y, corr);
  }
  return; 
}





DTMatchPt::DTMatchPt(string const s, int station,
		     const edm::ParameterSet& pSet, 
		     float const DTmu_x, float const DTmu_y,
		     float const stub_x[], float const stub_y[], 
		     bool const flagMatch[]):
  _label(s)  
{ 

  float corr = 0.0;
  //if(_label[0] == 'I') corr = 0.156; 
  //else if(_label[0] == 'M') corr = 0.05;

  float X[3], Y[3]; 

  _Rb = _invRb = NAN;
  _Pt = _invPt = NAN;
  _alpha0 = _d = NAN;

  if(isnan(DTmu_x) || isnan(DTmu_y)) 
    return;

  X[0] = DTmu_x;
  Y[0] = DTmu_y;
	
	if( _label == string("Mu_9_8") && flagMatch[5] && flagMatch[4] ) {
		X[1] = stub_x[5];
		Y[1] = stub_y[5];
		X[2] = stub_x[4];
		Y[2] = stub_y[4];
		computePt(pSet, X, Y, corr);
		return;
	}
	if( _label == string("Mu_9_3") && flagMatch[5] && flagMatch[3] ) {
		X[1] = stub_x[5];
		Y[1] = stub_y[5];
		X[2] = stub_x[3];
		Y[2] = stub_y[3];
		computePt(pSet, X, Y, corr);
		return;
	}
	if( _label == string("Mu_9_2") && flagMatch[5] && flagMatch[2] ) {
		X[1] = stub_x[5];
		Y[1] = stub_y[5];
		X[2] = stub_x[2];
		Y[2] = stub_y[2];
		computePt(pSet, X, Y, corr);
		return;
	}
	if( _label == string("Mu_9_1") && flagMatch[5] && flagMatch[1] ) {
		X[1] = stub_x[5];
		Y[1] = stub_y[5];
		X[2] = stub_x[1];
		Y[2] = stub_y[1];
		computePt(pSet, X, Y, corr);
		return;
	}
	if( _label == string("Mu_9_0") && flagMatch[5] && flagMatch[0] ) {
		X[1] = stub_x[5];
		Y[1] = stub_y[5];
		X[2] = stub_x[0];
		Y[2] = stub_y[0];
		computePt(pSet, X, Y, corr);
		return;
	}
	if( _label == string("Mu_8_3") && flagMatch[4] && flagMatch[3] ) {
		X[1] = stub_x[4];
		Y[1] = stub_y[4];
		X[2] = stub_x[3];
		Y[2] = stub_y[3];
		computePt(pSet, X, Y, corr);
		return;
	}
	if( _label == string("Mu_8_2") && flagMatch[4] && flagMatch[2] ) {
		X[1] = stub_x[4];
		Y[1] = stub_y[4];
		X[2] = stub_x[2];
		Y[2] = stub_y[2];
		computePt(pSet, X, Y, corr);
		return;
	}
	if( _label == string("Mu_8_1") && flagMatch[4] && flagMatch[1] ) {
		X[1] = stub_x[4];
		Y[1] = stub_y[4];
		X[2] = stub_x[1];
		Y[2] = stub_y[1];
		computePt(pSet, X, Y, corr);
		return;
	}
	if( _label == string("Mu_8_0") && flagMatch[4] && flagMatch[0] ) {
		X[1] = stub_x[4];
		Y[1] = stub_y[4];
		X[2] = stub_x[0];
		Y[2] = stub_y[0];
		computePt(pSet, X, Y, corr);
		return;
	}
  if( _label == string("Mu_3_2") && flagMatch[3] && flagMatch[2] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[2];
    Y[2] = stub_y[2];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_3_1") && flagMatch[3] && flagMatch[1] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[1];
    Y[2] = stub_y[1];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_3_0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_2_1") && flagMatch[2] && flagMatch[1] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[1];
    Y[2] = stub_y[1];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_2_0") && flagMatch[2] && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_1_0") && flagMatch[1] && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_9_V") && flagMatch[5] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_3_V") && flagMatch[3] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_2_V") && flagMatch[2] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_1_V") && flagMatch[1] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu_0_V") && flagMatch[0] ) {
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_9_0") && flagMatch[5] && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_3_2") && flagMatch[3] && flagMatch[2] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[2];
    Y[2] = stub_y[2];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_3_1") && flagMatch[3] && flagMatch[1] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[1];
    Y[2] = stub_y[1];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_3_0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_2_1") && flagMatch[3] && flagMatch[1] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[1];
    Y[2] = stub_y[1];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_2_0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_1_0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_9_V") && flagMatch[5] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_3_V") && flagMatch[3] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_2_V") && flagMatch[2] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_1_V") && flagMatch[1] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu_0_V") && flagMatch[0] ) {
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_9_0") && flagMatch[5] && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_3_2") && flagMatch[3] && flagMatch[2] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[2];
    Y[2] = stub_y[2];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_3_1") && flagMatch[3] && flagMatch[1] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[1];
    Y[2] = stub_y[1];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_3_0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_2_1") && flagMatch[2] && flagMatch[1] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[1];
    Y[2] = stub_y[1];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_2_0") && flagMatch[2] && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_1_0") && flagMatch[1] && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_9_V") && flagMatch[5] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_3_V") && flagMatch[3] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_2_V") && flagMatch[2] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_1_V") && flagMatch[2] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu_0_V") && flagMatch[0] ) {
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    X[2] = 0.;
    Y[2] = 0.;
    computePt(pSet, X, Y, corr);
    return;
  }
}





DTMatchPt::DTMatchPt(string const s, int station,
		     const edm::ParameterSet& pSet, 
		     const float vstub_rho, const float vstub_phi,
		     const GlobalVector stub_position[],
		     bool const flagMatch[]):
  _label(s)  
{ 
  /* 
  float corr = 0.0;
  //if(_label[0] == 'I') corr = 0.156; 
  //else if(_label[0] == 'M') corr = 0.05;

  GlobalVector P[2];

  _Rb = _invRb = _Pt = _invPt = NAN; 
  _alpha0 = _d = NAN;

  if(isnan(vstub_rho) || isnan(vstub_phi)) 
    return;
  
  if( _label == string("newestIMu_2_0") && flagMatch[2] && flagMatch[0] ) {
    P[1] = stub_position[2];
    P[0] = stub_position[0];
    computePt(pSet,  vstub_rho, vstub_phi, P,corr);
    return;
  }
  if( _label == string("newestIMu_3_0") && flagMatch[3] && flagMatch[0] ) {
    P[1] = stub_position[3];
    P[0] = stub_position[0];
    computePt(pSet, vstub_rho, vstub_phi, P, corr);
    return;
  }
 
  if( _label == string("newestIMu_3_V") && flagMatch[3] ) {
    P[1] = stub_position[3];
    P[0] = GlobalVector();
    computePt(pSet, vstub_rho, vstub_phi, P, corr);
    return;
  }
  if( _label == string("newestIMu_2_V") && flagMatch[3] ) {
    P[1] = stub_position[2];
    P[0] = GlobalVector();
    computePt(pSet, vstub_rho, vstub_phi, P, corr);
    return;
  }
  if( _label == string("newestIMu_1_V") && flagMatch[3] ) {
    P[1] = stub_position[1];
    P[0] = GlobalVector();
    computePt(pSet, vstub_rho, vstub_phi, P, corr);
    return;
  }
  if( _label == string("newestIMu_0_V") && flagMatch[0] ) {
    P[1] = stub_position[0];
    P[0] = GlobalVector();
    computePt(pSet, vstub_rho, vstub_phi, P, corr);
    return;
  }
  */
}





// using linear fit of stub dephi vs invPt
DTMatchPt::DTMatchPt(std::string const s, 
		     const float slope, const float dephi_zero,
		     const int I, const int J, const float dephi, 
		     const edm::ParameterSet& pSet, 
		     bool const flagMatch[]):
  _label(s) { 
  _Rb = _invRb = NAN;
  _Pt = _invPt = NAN;
  _alpha0 = _d = NAN;
  if( isnan(dephi) || isnan(dephi_zero) ) //|| dephi < 0.0015 ) 
    return;
  if( flagMatch[I] && flagMatch[J] && !isnan(slope) ) {
    _invPt = (dephi - dephi_zero)/slope;
    _Pt = slope/(dephi - dephi_zero);
  }
  return;
}
