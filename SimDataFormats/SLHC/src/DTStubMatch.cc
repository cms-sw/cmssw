#include <math.h>

//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatch.h"
#include "SimDataFormats/SLHC/interface/DTStubMatch.h"
#include "SLHCUpgradeSimulations/L1Trigger/interface/DTParameters.h"     

using namespace std;


int const our_to_tracker_lay_Id(int const l) {
  return StackedLayersInUse[l];
}


int const tracker_lay_Id_to_our(int const l) {
  int our = l;
  if(l == 8) our = 4;
  else if(l == 9) our = 5;
  return our;
}


void DTStubMatch::extrapolateToTrackerLayer(int l) {

  /**
     Argument "l" is "our" id for tracker layers in use: 6 (six) of them,
     namely 0,1,2,3,8,9.
  **/

  // ************************************************************************
  // *** Compute DT predicted phi, theta on tracker layers (each wheel)   ***
  // ************************************************************************
 
  // 090126 SV : compute Predicted in tracker layers from Ext 
  // NB needs to be extrapolated because of bending...
  // Ext = m Bending + q = Bti - Predicted, where Bti = Trigger + sector
  // Predicted = Trigger - (m Bending + q) + sector
  // Predicted for each tracker layer 0...5, from mb1 and mb2 stations.
  
  // PLZ (April 6 2009): Updated for longbarrel October 16 2009
  // extrapolation by wheel and digitization of extrapolated tracks and search windows.

  // MB1 parameters for tracker layers 0,1,2,3,8,9
  float m1[5][6] = {
  		   {12.08, 11.93, 11.57, 11.48, 10.06,  9.99},
                   {11.00, 10.88, 10.60, 10.28,  9.21,  9.15}, 
		   {11.21, 11.08, 10.78, 10.68,  9.45,  9.32}, 
		   {10.96, 10.84, 10.53, 10.44,  9.23,  9.12}, 
		   {12.15, 12.08, 11.65, 11.53, 10.10, 10.00} };
  float q1[5][6] = {
                   { 6.94,  6.84,  6.66,  6.56,  5.69,  5.65},
                   { 5.42,  5.35,  5.21,  5.15,  4.46,  4.42}, 
		   { 4.11,  4.07,  3.96,  3.91,  3.47,  3.42}, 
		   { 9.20,  9.09,  8.82,  8.74,  7.66,  7.56},
		   { 8.84,  8.79,  8.46,  8.35,  7.27,  7.19} };
  // MB2 parameters for tracker layers 0,1,2,3,4,5
  float m2[5][6] = {
                   {21.44, 21.18, 20.64, 20.34, 17.98, 17.74}, 
                   {18.12, 17.99, 17.51, 17.34, 15.34, 15.20}, 
		   {18.02, 17.85, 17.36, 17.19, 15.25, 15.13}, 
		   {18.26, 18.04, 17.57, 17.37, 15.43, 15.33}, 
		   {21.45, 21.26, 20.66, 20.45, 18.03, 17.89} };
  float q2[5][6] = {
                   {20.94, 20.68, 20.14, 19.85, 17.48, 17.24},
                   {16.49, 16.36, 15.91, 15.76, 13.89, 13.75},
		   { 9.78,  9.71,  9.44,  9.35,  8.29,  8.23},
		   {18.91, 18.67, 18.17, 17.96, 15.90, 15.79},
		   {20.95, 20.76, 20.16, 19.95, 17.53, 17.39} };

  // functions to compute extrapolation windows
  // MB1 parameters for computing errors on phi_ts and phib for each wheel
  float aphi1[5]=  {0.0032, 0.0018, 0.0008, 0.0021, 0.0012};
  float bphi1[5]=  {0.0365, 0.0131, 0.0394, 0.0056, 0.0473};
  float cphi1[5]=  {0.7117, 0.2325, 0.1314, 0.2581, 0.478};
  
  float aphib1[5]=  {0.0002776 , 0.00013179 , 0.0001983 , 0.0001303 , 0.0003104};
  float bphib1[5]=  {0.07319 , 0.06848 , 0.06304 , 0.06931 , 0.06933};
  float cphib1[5]=  {1.186 , 1.031 , 1.181 , 1.019 , 1.237};
  
  // MB2 parameters for computing errors on phi_ts and phib for each wheel  
  float aphi2[5]=  {0.0035, 0.0013, 0.0012, 0.0015, 0.0016};
  float bphi2[5]=  {0.0935, 0.0438, 0.0456, 0.0372, 0.0517};
  float cphi2[5]=  {2.606, 0.5139, 0.5092, 0.5693, 1.215};  
  
  float aphib2[5]=  {0.00243 , 0.0008163 , 0.001155 , 0.0008586 , 0.002531};
  float bphib2[5]=  {0.05192 , 0.06902 , 0.04317 , 0.06585 , 0.04321};
  float cphib2[5]=  {1.761 , 1.407 , 1.648 , 1.432 , 1.832};
  
  // 090209 SV : compute theta predicted: do not extrapolate: there is NO bending!
  // errors on theta (layer dependent, but not station nor pT dependent, 
  // for the moment --> FIX)
  float sigma_theta[6]={0.125,0.096,0.060,0.050,0.034,0.032};
  
  // Integer calculations: shift by 3 bits to preserve resolution
  int bit_shift = 8; 
  
  // station mb1
  if(station()==1 ){//&& flagBxOK() ){
    int iwh 	= wheel()+2;
    int phib1 	= phib_ts();
    int phi_ext = 
      static_cast<int>(m1[iwh][l]*bit_shift)*phib1 + static_cast<int>(q1[iwh][l]*bit_shift);
    // PLZ: extrapolation seems worse for code<16, try not to do it in this case 
    // SV 090505 FIXED: extrapolate for code<16 and take 5GeV bending cut
    //if(code() >= 16)
    int phi_mb1_track =	
      phi_ts() - phi_ext/bit_shift + static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
    //else 
    //  phi_mb1_track = phi_ts() 
    //			+ static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
    
    if(phi_mb1_track < 0)
      phi_mb1_track += static_cast<int>(2.*TMath::Pi()*4096.);
    
    // for sigma computation abs value is fine
    phib1 = abs(phib1);
    //	  if(code() < 16) phib1 = 70;		// 10 GeV cut	  
    if(code() < 16) phib1 = 160; 	// 5 GeV cut
    
    float sigma_phi  = aphi1[iwh]*phib1*phib1 + bphi1[iwh]*phib1 + cphi1[iwh];
    float sigma_phib  = aphib1[iwh]*phib1*phib1 + bphib1[iwh]*phib1 + cphib1[iwh];
    int sigma_phi_mb1_track = 
      static_cast<int>(sqrt(sigma_phi*sigma_phi+m1[iwh][l]*m1[iwh][l]*sigma_phib*sigma_phib));
    
    int theta_mb1_track= static_cast<int>(thetaCMS()*4096);
    int sigma_theta_mb1_track = static_cast<int>(sigma_theta[l]*4096);
    
    if(_debug_dttrackmatch) {
      cout << "Match at bx # " << bx() 
	   << " extrapolate MB1 to layer " << our_to_tracker_lay_Id(l) 
	   << " phiPredicted " << phi_mb1_track << " +- " << sigma_phi_mb1_track 
	   << " from phi Trigger " << phi_ts()
	   << " and phi Bending " << (phib_ts()) 
	   << " thetaPredicted " << theta_mb1_track << " +- " << sigma_theta_mb1_track 
	   << endl;
    }
    // store in DTStubMatch	
    // int lay = l; 
    // if(l > 7) lay = l-4;
    // Ignazio 091116 adopting tracker_lay_Id_to_our converter
    //int lay = DTStubMatch::tracker_lay_Id_to_our[l];
    setPredStubPhi(l,phi_mb1_track,sigma_phi_mb1_track);
    setPredStubTheta(l,theta_mb1_track,sigma_theta_mb1_track);
    setPredSigmaPhiB(sigma_phib);
  }// end MB1
  
  // station mb2
  if(station()==2 ){//&& flagBxOK() ){
    int iwh 	 = wheel()+2;
    int phib2 	 = phib_ts();
    int phi_ext = 
      static_cast<int>(m2[iwh][l]*bit_shift)*phib2 + 
      static_cast<int>(q2[iwh][l]*bit_shift);
    // PLZ: extrapolation seems worse for code<16, try not to do it in this case
    // SV 090505 FIXED: extrapolate for code<16 and take 5GeV bending cut
    //if(code() >= 16)
    int phi_mb2_track = 
      phi_ts() - phi_ext/bit_shift + 
      static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
    //else 
    //	phi_mb2_track = 	phi_ts() 
    //				+ static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
    if(phi_mb2_track < 0)
      phi_mb2_track += static_cast<int>(2.*TMath::Pi()*4096.);
    
    // for sigma computation abs value is fine
    phib2 = abs(phib2);
    //	  if(code() < 16) phib2 = 45;  // 10 GeV cut
    if(code() < 16) 
      phib2 = 85;  // 5 GeV cut
    
    float sigma_phi   = aphi2[iwh]*phib2*phib2 + bphi2[iwh]*phib2 + cphi2[iwh];
    float sigma_phib  = aphib2[iwh]*phib2*phib2 + bphib2[iwh]*phib2 + cphib2[iwh];
    int sigma_phi_mb2_track = 
      static_cast<int>(sqrt(sigma_phi*sigma_phi + 
			    m2[iwh][l]*m2[iwh][l]*sigma_phib*sigma_phib));
    
    int theta_mb2_track = static_cast<int>(thetaCMS()*4096);
    int sigma_theta_mb2_track = static_cast<int>(sigma_theta[l]*4096);
    
    if(_debug_dttrackmatch) { 
      cout << "Match at bx # " << bx() 
	   << " extrapolate MB2 to layer " << our_to_tracker_lay_Id(l)
	   << " phiPredicted " << phi_mb2_track << " +- " << sigma_phi_mb2_track 
	   << " from phi Trigger " << phi_ts() 
	   << " and phi Bending " << (phib_ts()) 
	   << " thetaPredicted " << theta_mb2_track << " +- " << sigma_theta_mb2_track 
	   << endl;
    }
    // store in DTStubMatch
    // int lay = l;
    // if(l > 7) lay = l-4;
    // Ignazio 091116 adopting tracker_lay_Id_to_our converter
    //int lay = DTStubMatch::tracker_lay_Id_to_our[l];	
    setPredStubPhi(l, phi_mb2_track, sigma_phi_mb2_track);
    setPredStubTheta(l, theta_mb2_track, sigma_theta_mb2_track);
    setPredSigmaPhiB(sigma_phib);
  }//end MB2
  
  return;
}



int DTStubMatch::corrPhiBend1ToCh2(int phib2) {
  // correlation function parameters for each wheel
  float a[5] = {5.E-5, 		1.E-5, 		2.E-5, 		2.E-5, 		5.E-5};
  float b[5] = {-0.0002,	6.E-5, 		-0.0001,	-7.E-05,	2.E-6};
  float c[5] = {1.4886, 	1.4084, 	1.3694, 	1.4039, 	1.4871};
  float d[5] = {0.7017, 	0.3776, 	0.6627, 	0.623, 		0.5025}; 

  // find phib in station 1 correlated with phib given for station 2
  int phib1 = 0;
  int iwh = wheel()+2;

  if(station()==1)
    phib1 = static_cast<int>(a[iwh]*phib2*phib2*phib2 + 
			     b[iwh]*phib2*phib2 + c[iwh]*phib2 + d[iwh]);
  else //no correlation, return the value in input
    phib1 = phib2;
  
  return phib1;
}



int DTStubMatch::corrSigmaPhiBend1ToCh2(int phib2, int sigma_phib2) {
  // correlation function parameters for each wheel
  float a[5] = {5.E-5, 		1.E-5, 		2.E-5, 		2.E-5, 		5.E-5};
  float b[5] = {-0.0002,	6.E-5, 		-0.0001,	-7.E-05,	2.E-6};
  float c[5] = {1.4886, 	1.4084, 	1.3694, 	1.4039, 	1.4871};
  //float d[5] = {0.7017, 	0.3776, 	0.6627, 	0.623, 		0.5025}; 
  
  // find phib error in station 1 correlated with phib given for station 2
  int sigma_phib1 = 0;
  int iwh = wheel()+2;
  
  if(station()==1)
    sigma_phib1 = static_cast<int>(fabs((3*a[iwh]*phib2*phib2 + 
					 2*b[iwh]*phib2 + c[iwh]*phib2) * sigma_phib2));
  else //no correlation, return the value in input
    sigma_phib1 = sigma_phib2;
  
  return sigma_phib1;
}



void DTStubMatch::print() {

  cout 	<< "DTStubMatch : wh " << wheel() << ", st " << station() << ", se " << sector() 
	<< ", bx " << bx() << ", code " << code() << " rejection " << flagReject() 
	<< endl;  
}







//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Ignazio
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void DTStubMatch::init() {

  _trig_order = -555555555;
  _pred_theta = -555555555;
  _pred_sigma_phib = NAN;

  _erre = _erresq = NAN;
  _xerre = _Xerre = _XerreI = NAN;
  _yerre = _Yerre = _YerreI = NAN;

  _PhiR = _phiR = NAN;
  _deltaPhiR = NAN;
  _deltaPhiR_over_bendingDT = NAN;
  _deltaPhiR_over_bendingDT_S1 = NAN;
  _deltaPhiR_over_bendingDT_S2 = NAN;
  _deltaPhiR_over_bendingDT_S1_0 = NAN; 
  _deltaPhiR_over_bendingDT_S1_1 = NAN;
  _deltaPhiR_over_bendingDT_S1_2 = NAN;
  _deltaPhiR_over_bendingDT_S2_0 = NAN; 
  _deltaPhiR_over_bendingDT_S2_1 = NAN;
  _deltaPhiR_over_bendingDT_S2_2 = NAN;

  _deltaPhiL9_over_bendingDT = NAN;

  _sqrtDscrm = NAN;
  
  for(int l=0; l<StackedLayersInUseTotal; l++){	
    _pred_phi[l] = -555555555;
    _pred_sigma_phi[l] = -555555555;
    _pred_sigma_theta[l] = -555555555;
    _stub_phi[l] = -555555555;
    _stub_theta[l] = -555555555;
    _stub_x[l] = NAN;
    _stub_y[l] = NAN;
    _stub_z[l] = NAN;
    _stub_rho[l]= NAN;
    _stub_direction[l] = GlobalVector();
    _flagMatch[l] = false;

  }
  //----------------------------------------------------------------------//

  _matching_stubs = StubTracklet();  
  _matching_stubs_No = 0;

  Stubs_5_3_0 = DTStubMatchPt(std::string("Stubs-5-3-0"));
  Stubs_5_1_0 = DTStubMatchPt(std::string("Stubs-5-1-0"));
  Stubs_3_2_0 = DTStubMatchPt(std::string("Stubs-3-2-0")); 
  Stubs_3_1_0 = DTStubMatchPt(std::string("Stubs-3-1-0")); 
  Stubs_5_3_V = DTStubMatchPt(std::string("Stubs-5-3-V")); 
  Stubs_5_0_V = DTStubMatchPt(std::string("Stubs-5-0-V")); 
  Stubs_3_0_V = DTStubMatchPt(std::string("Stubs-3-0-V"));
  Mu_5_0   = DTStubMatchPt(std::string("Mu-5-0"));
  Mu_3_0   = DTStubMatchPt(std::string("Mu-3-0"));
  Mu_2_0   = DTStubMatchPt(std::string("Mu-2-0"));
  Mu_1_0   = DTStubMatchPt(std::string("Mu-1-0"));
  Mu_5_V   = DTStubMatchPt(std::string("Mu-5-V"));
  Mu_3_V   = DTStubMatchPt(std::string("Mu-3-V"));
  Mu_2_V   = DTStubMatchPt(std::string("Mu-3-V"));
  Mu_1_V   = DTStubMatchPt(std::string("Mu-1-V"));
  Mu_0_V   = DTStubMatchPt(std::string("Mu-0-V"));
  IMu_5_0   = DTStubMatchPt(std::string("IMu-5-0"));
  IMu_3_0   = DTStubMatchPt(std::string("IMu-3-0"));
  IMu_2_0   = DTStubMatchPt(std::string("IMu-2-0"));
  IMu_1_0   = DTStubMatchPt(std::string("IMu-1-0"));
  IMu_5_V   = DTStubMatchPt(std::string("IMu-5-V"));
  IMu_3_V   = DTStubMatchPt(std::string("IMu-3-V"));
  IMu_2_V   = DTStubMatchPt(std::string("IMu-3-V"));
  IMu_1_V   = DTStubMatchPt(std::string("IMu-1-V"));
  IMu_0_V   = DTStubMatchPt(std::string("IMu-0-V"));
  mu_5_0  = DTStubMatchPt(std::string("mu-5-0"));
  mu_3_0  = DTStubMatchPt(std::string("mu-3-0"));
  mu_2_0  = DTStubMatchPt(std::string("mu-2-0"));
  mu_1_0  = DTStubMatchPt(std::string("mu-1-0"));
  mu_5_V  = DTStubMatchPt(std::string("mu-5-V"));
  mu_3_V  = DTStubMatchPt(std::string("mu-3-V"));
  mu_2_V  = DTStubMatchPt(std::string("mu-2-V"));
  mu_1_V  = DTStubMatchPt(std::string("mu-1-V"));
  mu_0_V  = DTStubMatchPt(std::string("mu-0-V"));
  only_Mu_V =  DTStubMatchPt(std::string("only-Mu-V"));

  //-----------------------------------------------------------------------//
   _alphaDT = _Xerre = _Yerre = _xerre = _yerre = NAN;
 
  _matching_stubs_No = 0;
  
  _flag_reject = false;
}



DTStubMatch::DTStubMatch() {

  _wheel     = -9999999;
  _station   = -9999999;
  _sector    = -9999999;
  _bx        = -9999999;
  _code      = -9999999;
  _phi_ts    = -9999999;
  _phib_ts   = -9999999;
  _theta_ts  = -9999999;
  _position  = GlobalPoint();
  _direction = GlobalVector();
  _rhoDT     = NAN;
  _phiCMS    = NAN;
  _bendingDT = NAN;
  _thetaCMS  = NAN;
  _eta       = NAN;

  _flagBxOK  = false;
  
  init();

}



DTStubMatch::DTStubMatch(int wheel, int station, int sector,
			 int bx, int code, int phi, int phib, float theta, bool flagBxOK,
			 bool debug_dttrackmatch) {

  init();

  _wheel = wheel;
  _station = station;
  _sector = sector;
  _bx = bx;
  _code = code;
  _phi_ts = phi;
  _phib_ts = phib;
  _theta_ts  = -9999999;  /// !!!!!!!!!!!! rimediare!
  _thetaCMS = theta;    
  _eta = -log(tan(theta/2.));
  _position  = GlobalPoint();
  _direction = GlobalVector();
  _flagBxOK  = flagBxOK;
  _rhoDT     = NAN;
  _phiCMS    = this->phi_glo();
  _bendingDT = this->phib_glo();
  _debug_dttrackmatch = debug_dttrackmatch;   

}


// constructor
DTStubMatch::DTStubMatch(int wheel, int station, int sector,
			 int bx, int code, int phi, int phib, float theta, 
			 GlobalPoint position, GlobalVector direction,
			 bool flagBxOK, bool debug_dttrackmatch) {

  init();

  _wheel = wheel;
  _station = station;
  _sector = sector;
  _bx = bx;
  _code = code;
  _phi_ts = phi;
  _phib_ts = phib;
  _theta_ts  = -9999999;  
  /// !!!!!!!!!!!! rimediare! usare codice commentato in DTSimTrigger.cc
  _thetaCMS = theta;
  _eta = -log(tan(theta/2.));
  _position  = position;
  _direction = direction;
  _phiCMS    = this->phi_glo();
  _bendingDT = this->phib_glo();

  _flagBxOK = flagBxOK;  

  _debug_dttrackmatch = debug_dttrackmatch;


  double delta = static_cast<double>(_bendingDT); 

  _alphaDT = _phiCMS + delta;
  if(_alphaDT < 0.)
    _alphaDT += 2 * TMath::Pi();
  if(_alphaDT > 2*TMath::Pi())
    _alphaDT -= 2 * TMath::Pi();

  // first approach ------------------------------------------------------------ 
  double rhoDTsq = _position.x()*_position.x() + _position.y()*_position.y();
  double rhoDT   = sqrt(rhoDTsq);
  _rhoDT = static_cast<float>(rhoDT);
  double Dscrm  = 1. - rhoDTsq*sin(delta)*sin(delta)/(Erre*Erre);
  if( Dscrm < 0.)  
    return;

  _sqrtDscrm = static_cast<float>(sqrt(Dscrm));

  _xerre = Erre*cos(_alphaDT)*_sqrtDscrm + rhoDT*sin(delta)*sin(_alphaDT);
  _yerre = Erre*sin(_alphaDT)*_sqrtDscrm - rhoDT*sin(delta)*cos(_alphaDT);

  // Just to have a feeling....
  _phiR = (_yerre>0.)? acos(_xerre/Erre): (2.*TMath::Pi() - acos(_xerre/Erre));
  if(_phiR < 0.)
    _phiR += 2 * TMath::Pi();
  if(_phiR > 2*TMath::Pi())
    _phiR -= 2 * TMath::Pi();
  _deltaPhiR = _phiR - _phiCMS; 
  _deltaPhiR_over_bendingDT =(fabs(delta) < 0.000001)? NAN: _deltaPhiR/delta;
  if( _station == 1 ) 
    _deltaPhiR_over_bendingDT_S1 = _deltaPhiR_over_bendingDT;
  else if( _station == 2 )
    _deltaPhiR_over_bendingDT_S2 = _deltaPhiR_over_bendingDT;
  if( _station == 1 && _wheel == 0)
    _deltaPhiR_over_bendingDT_S1_0 = _deltaPhiR_over_bendingDT;
  else if( _station == 1 && abs(_wheel) == 1)
    _deltaPhiR_over_bendingDT_S1_1 = _deltaPhiR_over_bendingDT;
  else if( _station == 1 && abs(_wheel) == 2)
    _deltaPhiR_over_bendingDT_S1_2 = _deltaPhiR_over_bendingDT;
  else if( _station == 2 && _wheel == 0)
    _deltaPhiR_over_bendingDT_S2_0 = _deltaPhiR_over_bendingDT;
  else if( _station == 2 && abs(_wheel) == 1)
    _deltaPhiR_over_bendingDT_S2_1 = _deltaPhiR_over_bendingDT;
  else if( _station == 2 && abs(_wheel) == 2)
    _deltaPhiR_over_bendingDT_S2_2 = _deltaPhiR_over_bendingDT;
  // ... end of just to have a feeling.


  // second approach ------------------------------------------------------------
  _Xerre = Erre*cos(_alphaDT) + rhoDT*sin(_alphaDT)*delta;
  _Yerre = Erre*sin(_alphaDT) - rhoDT*cos(_alphaDT)*delta;
  

  // third approach -------------------------------------------------------------
  static float deltaPhiR_over_bendingDT  = 1. - _rhoDT/Erre;
  static float deltaPhiR = deltaPhiR_over_bendingDT * delta; 
  _PhiR = _phiCMS + deltaPhiR;
  if(_PhiR < 0.)
    _PhiR += 2 * TMath::Pi();
  if(_PhiR > 2*TMath::Pi())
    _PhiR -= 2 * TMath::Pi();
  _XerreI = Erre*cos(_PhiR);
  _YerreI = Erre*sin(_PhiR);

} 





// copy constructor
DTStubMatch::DTStubMatch(const DTStubMatch& dtsm) {
  _wheel = dtsm.wheel();
  _station = dtsm.station(); 
  _sector = dtsm.sector(); 
  _bx = dtsm.bx();
  _code = dtsm.code();
  _phi_ts = dtsm.phi_ts(); 
  _phib_ts = dtsm.phib_ts();
  _phiCMS = dtsm.phiCMS();
  _bendingDT = dtsm._bendingDT; 
  _thetaCMS = dtsm.thetaCMS();
  _eta = dtsm.eta();
  _flagBxOK = dtsm.flagBxOK(); 
  _trig_order = dtsm.trig_order();
  _flag_reject = dtsm.flagReject();

  _position = dtsm.position();
  _direction = dtsm.direction();

  _alphaDT = dtsm.alphaDT();
  _sqrtDscrm = dtsm.sqrtDiscrim();
  _Xerre = dtsm.Xerre();
  _Yerre = dtsm.Yerre();
  _xerre = dtsm.xerre();
  _yerre = dtsm.yerre();
  _XerreI = dtsm.XerreI();
  _YerreI = dtsm.YerreI();
  _rhoDT = dtsm.rhoDT();
  _phiCMS = dtsm.phiCMS();

  _PhiR = dtsm._PhiR;
  _phiR = dtsm._phiR;
  _deltaPhiR = dtsm._deltaPhiR;
  _deltaPhiR_over_bendingDT_S1 = dtsm._deltaPhiR_over_bendingDT_S1;
  _deltaPhiR_over_bendingDT_S2 = dtsm._deltaPhiR_over_bendingDT_S2;
  _deltaPhiR_over_bendingDT_S1_0 = dtsm._deltaPhiR_over_bendingDT_S1_0;
  _deltaPhiR_over_bendingDT_S1_1 = dtsm._deltaPhiR_over_bendingDT_S1_1;
  _deltaPhiR_over_bendingDT_S1_2 = dtsm._deltaPhiR_over_bendingDT_S1_2;
  _deltaPhiR_over_bendingDT_S2_0 = dtsm._deltaPhiR_over_bendingDT_S2_0;
  _deltaPhiR_over_bendingDT_S2_1 = dtsm._deltaPhiR_over_bendingDT_S2_1;
  _deltaPhiR_over_bendingDT_S2_2 = dtsm._deltaPhiR_over_bendingDT_S2_2;

  _deltaPhiL9_over_bendingDT = dtsm._deltaPhiL9_over_bendingDT;

  _matching_stubs = dtsm.getMatchingStubs();
  _matching_stubs_No = _matching_stubs.size();

  _pred_theta = dtsm.predTheta();
  _pred_sigma_phib = dtsm.predSigmaPhiB();

  for(int l=0; l<StackedLayersInUseTotal; l++) {
    _pred_phi[l] = dtsm.predPhi(l);
    _pred_sigma_phi[l] = dtsm.predSigmaPhi(l);
    _pred_sigma_theta[l] = dtsm.predSigmaTheta(l);
    _stub_phi[l] = dtsm.stubPhi(l);
    _stub_theta[l] = dtsm.stubTheta(l);
    _flagMatch[l] = dtsm.isMatched(l);
    _stub_x[l]  = dtsm._stub_x[l];
    _stub_y[l] = dtsm._stub_y[l]; 
    _stub_z[l] = dtsm._stub_z[l]; 
    _stub_rho[l] = dtsm._stub_rho[l]; 
    _stub_direction[l] = dtsm._stub_direction[l];
  }

  Stubs_5_3_0 = DTStubMatchPt(dtsm.Stubs_5_3_0);
  Stubs_5_1_0 = DTStubMatchPt(dtsm.Stubs_5_1_0);
  Stubs_3_2_0 = DTStubMatchPt(dtsm.Stubs_3_2_0); 
  Stubs_3_1_0 = DTStubMatchPt(dtsm.Stubs_3_1_0); 
  Stubs_5_3_V = DTStubMatchPt(dtsm.Stubs_5_3_V); 
  Stubs_5_0_V = DTStubMatchPt(dtsm.Stubs_5_0_V); 
  Stubs_3_0_V = DTStubMatchPt(dtsm.Stubs_3_0_V);
  Mu_5_0   = DTStubMatchPt(dtsm.Mu_5_0);
  Mu_3_0   = DTStubMatchPt(dtsm.Mu_3_0);
  Mu_2_0   = DTStubMatchPt(dtsm.Mu_2_0);
  Mu_1_0   = DTStubMatchPt(dtsm.Mu_1_0);
  Mu_5_V   = DTStubMatchPt(dtsm.Mu_5_V);
  Mu_3_V   = DTStubMatchPt(dtsm.Mu_3_V);
  Mu_2_V   = DTStubMatchPt(dtsm.Mu_2_V);
  Mu_1_V   = DTStubMatchPt(dtsm.Mu_1_V);
  Mu_0_V   = DTStubMatchPt(dtsm.Mu_0_V);
  IMu_5_0   = DTStubMatchPt(dtsm.IMu_5_0);
  IMu_3_0   = DTStubMatchPt(dtsm.IMu_3_0);
  IMu_2_0   = DTStubMatchPt(dtsm.IMu_2_0);
  IMu_1_0   = DTStubMatchPt(dtsm.IMu_1_0);
  IMu_5_V   = DTStubMatchPt(dtsm.IMu_5_V);
  IMu_3_V   = DTStubMatchPt(dtsm.IMu_3_V);
  IMu_2_V   = DTStubMatchPt(dtsm.IMu_2_V);
  IMu_1_V   = DTStubMatchPt(dtsm.IMu_1_V);
  IMu_0_V   = DTStubMatchPt(dtsm.IMu_0_V);
  mu_5_0   = DTStubMatchPt(dtsm.mu_5_0);
  mu_3_0   = DTStubMatchPt(dtsm.mu_3_0);
  mu_2_0   = DTStubMatchPt(dtsm.mu_2_0);
  mu_1_0   = DTStubMatchPt(dtsm.mu_1_0);
  mu_5_V   = DTStubMatchPt(dtsm.mu_5_V);
  mu_2_V   = DTStubMatchPt(dtsm.mu_2_V);
  mu_1_V   = DTStubMatchPt(dtsm.mu_1_V);
  mu_0_V   = DTStubMatchPt(dtsm.mu_0_V);
  only_Mu_V = DTStubMatchPt(dtsm.only_Mu_V);
  
  _debug_dttrackmatch = dtsm._debug_dttrackmatch;
   
}
 
  

std::string DTStubMatch::writeMatchingStubs() const {
  // Ignazio
  std::ostringstream outString;
  StubTracklet::const_iterator st = _matching_stubs.begin();
  for(size_t i = 0; i<_matching_stubs.size(); i++) {
    outString << (*st)->id() << endl;
    ++st;
  }
  return outString.str();
}    


std::string DTStubMatch::writeMatchingStubs(size_t d) const {
  // Ignazio
  std::ostringstream outString;
  outString << "DTStubMatch " << d << " total of matchingStubs " 
	    << _matching_stubs.size() << "\n"; 
  StubTracklet::const_iterator st = _matching_stubs.begin();
  for(size_t i = 0; i<_matching_stubs.size(); i++) {
    outString << (*st)->id() << endl;
    ++st;
  }
  return outString.str();
}    



// Pt ***********************************************************************

void DTStubMatch::setPt(const edm::ParameterSet& pSet) 
{
  int i = -1;
  for(int l=0; l<StackedLayersInUseTotal; l++) {
    if( !_flagMatch[l] ) 
      continue;
    ++i;
  }
  _matching_stubs_No = i+1;
  vector<string> labels = 
    pSet.getUntrackedParameter<std::vector<std::string> >("labels");
  for(size_t s=0; s<labels.size(); s++) {
    DTStubMatchPt* aPt = new DTStubMatchPt();
    if((labels[s]) == string("only-Mu-V")) {
      float rB = (0.5 * Erre* Erre)/(_rhoDT*fabs(_bendingDT));
      only_Mu_V = DTStubMatchPt(_station, pSet, _bendingDT, rB);
    }
    else if((labels[s])[0] == 'm') 
      aPt = new DTStubMatchPt(labels[s], _station, pSet, 
			      _xerre, _yerre, _stub_x, _stub_y, _flagMatch); 
    else if((labels[s])[0] == 'M') 
      aPt = new DTStubMatchPt(labels[s], _station, pSet, 
			      _Xerre, _Yerre, _stub_x, _stub_y, _flagMatch);
    else if((labels[s])[0] == 'I') 
      aPt = new DTStubMatchPt(labels[s], _station, pSet, 
			      _XerreI, _YerreI, _stub_x, _stub_y, _flagMatch);
    else if((labels[s])[0] == 'S') 
      aPt = new DTStubMatchPt(labels[s], _station, pSet, _stub_x, _stub_y, _flagMatch);
  
    if(labels[s] == std::string("Stubs-5-3-0")) 
      Stubs_5_3_0 = DTStubMatchPt(*aPt);
    if(labels[s] == std::string("Stubs-5-1-0")) 
      Stubs_5_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-3-2-0")) 
      Stubs_3_2_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-3-1-0")) 
      Stubs_3_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-5-3-V")) 
      Stubs_5_3_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-5-0-V")) 
      Stubs_5_0_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-3-0-V")) 
      Stubs_3_0_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-5-0")) 
      Mu_5_0 = DTStubMatchPt(*aPt);  
    else if(labels[s] == std::string("Mu-3-0")) 
      Mu_3_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-2-0")) 
      Mu_2_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-1-0")) 
      Mu_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-5-V")) 
      Mu_5_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-3-V")) 
      Mu_3_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-2-V")) 
      Mu_2_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-1-V")) 
      Mu_1_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-0-V")) 
      Mu_0_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-5-0")) 
      IMu_5_0 = DTStubMatchPt(*aPt);  
    else if(labels[s] == std::string("IMu-3-0")) 
      IMu_3_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-2-0")) 
      IMu_2_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-1-0")) 
      IMu_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-5-V")) 
      IMu_5_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-3-V")) 
      IMu_3_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-2-V")) 
      IMu_2_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-1-V")) 
      IMu_1_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-0-V")) 
      IMu_0_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-5-0")) 
      mu_5_0 = DTStubMatchPt(*aPt);  
    else if(labels[s] == std::string("mu-3-0")) 
      mu_3_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-2-0")) 
      mu_2_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-1-0")) 
      mu_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-5-V")) 
      mu_5_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-3-V")) 
      mu_3_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-2-V")) 
      mu_2_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-1-V")) 
      mu_1_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-0-V")) 
      mu_0_V = DTStubMatchPt(*aPt);
  }
}



float const DTStubMatch::Pt(std::string const label) const
{
  if(label == std::string("Stubs-5-3-0")) 
    return Stubs_5_3_0.Pt();
  else if(label == std::string("Stubs-5-1-0")) 
    return Stubs_5_1_0.Pt();
  else if(label == std::string("Stubs-3-2-0")) 
    return Stubs_3_2_0.Pt();
  else if(label == std::string("Stubs-3-1-0")) 
    return Stubs_3_1_0.Pt();
  else if(label == std::string("Stubs-5-3-V")) 
    return Stubs_5_3_V.Pt();
  else if(label == std::string("Stubs-5-0-V")) 
    return Stubs_5_0_V.Pt(); 
  else if(label == std::string("Stubs-3-0-V")) 
    return Stubs_3_0_V.Pt();
  else if(label == std::string("Mu-5-0")) 
    return Mu_5_0.Pt();
  else if(label == std::string("Mu-3-0")) 
    return Mu_3_0.Pt();
  else if(label == std::string("Mu-2-0")) 
    return Mu_2_0.Pt();
  else if(label == std::string("Mu-1-0")) 
    return Mu_1_0.Pt();
  else if(label == std::string("Mu-5-V")) 
    return Mu_5_V.Pt();
  else if(label == std::string("Mu-3-V")) 
    return Mu_3_V.Pt();
  else if(label == std::string("Mu-2-V")) 
    return Mu_2_V.Pt();
  else if(label == std::string("Mu-1-V")) 
    return Mu_1_V.Pt();
  else if(label == std::string("Mu-0-V")) 
    return Mu_0_V.Pt();
  else if(label == std::string("IMu-5-0")) 
    return IMu_5_0.Pt();
  else if(label == std::string("IMu-3-0")) 
    return IMu_3_0.Pt();
  else if(label == std::string("IMu-2-0")) 
    return IMu_2_0.Pt();
  else if(label == std::string("IMu-1-0")) 
    return IMu_1_0.Pt();
  else if(label == std::string("IMu-5-V")) 
    return IMu_5_V.Pt();
  else if(label == std::string("IMu-3-V")) 
    return IMu_3_V.Pt();
  else if(label == std::string("IMu-2-V")) 
    return IMu_2_V.Pt();
  else if(label == std::string("IMu-1-V")) 
    return IMu_1_V.Pt();
  else if(label == std::string("IMu-0-V")) 
    return IMu_0_V.Pt();
  else if(label == std::string("mu-5-0")) 
    return mu_5_0.Pt();
  else if(label == std::string("mu-3-0")) 
    return mu_3_0.Pt();
  else if(label == std::string("mu-2-0")) 
    return mu_2_0.Pt();
  else if(label == std::string("mu-1-0")) 
    return mu_1_0.Pt();
  else if(label == std::string("mu-5-V")) 
    return mu_5_V.Pt();
  else if(label == std::string("mu-3-V")) 
    return mu_3_V.Pt();
  else if(label == std::string("mu-2-V")) 
    return mu_2_V.Pt();
  else if(label == std::string("mu-1-V")) 
    return mu_1_V.Pt();
  else if(label == std::string("mu-0-V")) 
    return mu_0_V.Pt();
  else if(label == std::string("only_Mu_V"))
    return only_Mu_V.Pt();
  return NAN;
}



//--------------------------------------------------------------------------
bool DTStubMatchSortPredicate(const DTStubMatch* d1, const DTStubMatch* d2)
{
  return (d1->trig_order() < d2->trig_order());
}

