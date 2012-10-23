#include <math.h>

#include "SimDataFormats/SLHC/interface/DTMatch.h"

using namespace std;


//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Ignazio
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

bool DTMatchSortPredicate(const DTMatch* d1, const DTMatch* d2)
{
  return (d1->trig_order() < d2->trig_order());
}


//-------------------------------------------------------------------------------

ostream& operator <<(ostream &os, const DTMatch &obj)
{
  for(size_t i=0; i<(RTSdataSize-1); i++)
    os << obj.RTSdata(i) << "  ";
  os << obj.RTSdata((RTSdataSize-1)) << endl;
  return os;
}




//------------------------------------------------------------------------------
void DTMatch::init() 
{
  _GunFiredSingleMuPt = 0.;
  _Pt_encoder = NAN;    // Ignazio
  _Pt_encoder_bin = NAN;      // Ignazio
  _Pt_average_bin = NAN;      // Ignazio
  _Pt_average_bin_Tracklet = NAN;      // Ignazio
  _Pt_majority_bin = NAN;      // Ignazio
  _Pt_majority_bin_Tracklet = NAN;      // PLZ
  _Pt_majority_bin_Full = NAN;      // PLZ
  _trig_order = -555555555;
  _pred_theta = -555555555;
  _pred_sigma_phib = NAN;
  _delta_theta = -555555555;
  
  for(int l=0; l<StackedLayersInUseTotal; l++){	
    _pred_phi[l] = -555555555;
    _pred_sigma_phi[l] = -555555555;
    _pred_sigma_theta[l] = -555555555;
  }
  
  _matching_stubs = DTMatchingStubSet();  
  _flag_reject = false;
  
  for(size_t i=0; i<RTSdataSize; i++)
    _RTSdata[i] = 0;
  /* 
     station
     sector
     phib_ts
     6 * ( phi, dist_phi )
     Pt_generated (in case!)
  */
}





DTMatch::DTMatch() {
  init();
  _phi_ts    = -9999999;
  _phib_ts   = -9999999;
  _theta_ts  = -9999999;
  _position  = GlobalPoint();
  _direction = GlobalVector();
  _flagBxOK  = false;  
  _flagPt    = false;
}




// constructor
DTMatch::DTMatch(int wheel, int station, int sector,
			 int bx, int code, int phi, int phib, float theta, 
			 bool flagBxOK,
			 bool debug_dttrackmatch) 
{
  init();
  _wheel = wheel;
  _station = station;
  _sector = sector;
  _bx = bx;
  _code = code;
  _phi_ts = phi;
  _phib_ts = phib;
  _theta_ts  = static_cast<int>(theta*4096.);  /// !!!!!!!!!!!! rimediare!
  _thetaCMS = theta;    
  _eta = -log(tan(theta/2.));
  _position  = GlobalPoint();
  _direction = GlobalVector();
  _flagBxOK  = flagBxOK;
  _phiCMS    = this->phi_glo();
  _bendingDT = this->phib_glo();
  _debug_dttrackmatch = debug_dttrackmatch; 
  _flagPt    = false;   
  _flag_theta = true;
  _RTSdata[0] = static_cast<short>(station);
  _RTSdata[1] = static_cast<short>(sector);
  _RTSdata[2] = static_cast<short>(phib);   
}





// constructor
DTMatch::DTMatch(int wheel, int station, int sector,
			 int bx, int code, int phi, int phib, float theta, 
			 GlobalPoint position, GlobalVector direction,
			 bool flagBxOK, bool debug_dttrackmatch) 
{
  init();
  _wheel = wheel;
  _station = station;
  _sector = sector;
  _bx = bx;
  _code = code;
  _phi_ts = phi;
  _phib_ts = phib;
  _theta_ts  = static_cast<int>(theta*4096.);  
  /// !!!!!!!!!!!! rimediare! usare codice commentato in DTSimTrigger.cc
  _thetaCMS = theta;
  _eta = -log(tan(theta/2.));
  _position  = position;
  _direction = direction;
  _phiCMS    = this->phi_glo();
  _bendingDT = this->phib_glo();

  _flagBxOK = flagBxOK;  
  _flagPt   = false; 
  _flag_theta = true;

  _RTSdata[0] = static_cast<short>(station);
  _RTSdata[1] = static_cast<short>(sector);
  _RTSdata[2] = static_cast<short>(phib);   

  _debug_dttrackmatch = debug_dttrackmatch;

  return;
} 





// copy constructor
DTMatch::DTMatch(const DTMatch& dtsm): DTMatchPtAlgorithms(dtsm) 
{
  // cout << "DTMatch copy constructor called" << endl;
  _phi_ts = dtsm.phi_ts(); 
  _phib_ts = dtsm.phib_ts();
  _flagBxOK = dtsm.flagBxOK(); 
  _trig_order = dtsm.trig_order();
  _flag_reject = dtsm.flagReject();
  _flagPt = dtsm.flagPt(); 
  _flag_theta = dtsm.flagTheta();
  _delta_theta = dtsm.deltaTheta();
  _Pt_encoder = dtsm.Pt_encoder();
  _Pt_average = dtsm.Pt_average();
  _Pt_encoder_bin = dtsm.Pt_encoder_bin();
  _Pt_average_bin = dtsm.Pt_average_bin();
  _Pt_average_bin_Tracklet = dtsm.Pt_average_bin_Tracklet();
  _Pt_majority_bin = dtsm.Pt_majority_bin();
  _Pt_majority_bin_Tracklet = dtsm.Pt_majority_bin_Tracklet();
  _Pt_majority_bin_Full = dtsm.Pt_majority_bin_Full();

  _matching_stubs = dtsm.getMatchingStubs();
  _matching_stubs_No = _matching_stubs.size();

  _pred_theta = dtsm.predTheta();
  _pred_sigma_phib = dtsm.predSigmaPhiB();

  for(int l=0; l<StackedLayersInUseTotal; l++) {
    _pred_phi[l] = dtsm.predPhi(l);
    _pred_sigma_phi[l] = dtsm.predSigmaPhi(l);
    _pred_sigma_theta[l] = dtsm.predSigmaTheta(l);
    _flagMatch[l] = dtsm.isMatched(l);
    _stub_x[l]  = dtsm._stub_x[l];
    _stub_y[l] = dtsm._stub_y[l]; 
    _stub_z[l] = dtsm._stub_z[l]; 
    _stub_rho[l] = dtsm._stub_rho[l]; 
    _stub_phiCMS[l] = dtsm._stub_phiCMS[l];
    _stub_position[l]  = dtsm._stub_position[l];
    _stub_direction[l] = dtsm._stub_direction[l];
  }
  for(size_t i=0; i< RTSdataSize; i++)
    _RTSdata[i] = dtsm._RTSdata[i];
  _debug_dttrackmatch = dtsm._debug_dttrackmatch;
  // cout << "DTMatch copy constructor done" << endl; 
}
 
  




// assignment operator
DTMatch& DTMatch::operator =(const DTMatch& dtsm) 
{
  // cout << "DTMatch assignment operator called" << endl;
  if (this == &dtsm)      // Same object?
    return *this;         // Yes, so skip assignment, and just return *this.
  this->DTMatchPtAlgorithms::operator=(dtsm);
  _phi_ts = dtsm.phi_ts(); 
  _phib_ts = dtsm.phib_ts();
  _flagBxOK = dtsm.flagBxOK(); 
  _trig_order = dtsm.trig_order();
  _flag_reject = dtsm.flagReject();
  _flagPt = dtsm.flagPt(); 
  _flag_theta = dtsm.flagTheta();
  _delta_theta = dtsm.deltaTheta();
  _Pt_encoder = dtsm.Pt_encoder();
  _Pt_average = dtsm.Pt_average();
  _Pt_encoder_bin = dtsm.Pt_encoder_bin();
  _Pt_average_bin = dtsm.Pt_average_bin();
  _Pt_average_bin_Tracklet = dtsm.Pt_average_bin_Tracklet();
  _Pt_majority_bin = dtsm.Pt_majority_bin();
  _Pt_majority_bin_Tracklet = dtsm.Pt_majority_bin_Tracklet();
  _Pt_majority_bin_Full = dtsm.Pt_majority_bin_Full();

  _matching_stubs = dtsm.getMatchingStubs();
  _matching_stubs_No = _matching_stubs.size();

  _pred_theta = dtsm.predTheta();
  _pred_sigma_phib = dtsm.predSigmaPhiB();

  for(int l=0; l<StackedLayersInUseTotal; l++) {
    _pred_phi[l] = dtsm.predPhi(l);
    _pred_sigma_phi[l] = dtsm.predSigmaPhi(l);
    _pred_sigma_theta[l] = dtsm.predSigmaTheta(l);
    _stub_x[l]  = dtsm._stub_x[l];
    _stub_y[l] = dtsm._stub_y[l]; 
    _stub_z[l] = dtsm._stub_z[l]; 
    _stub_rho[l] = dtsm._stub_rho[l]; 
    _stub_phiCMS[l] = dtsm._stub_phiCMS[l];
    _stub_position[l]  = dtsm._stub_position[l];
    _stub_direction[l] = dtsm._stub_direction[l];
  }
  for(size_t i=0; i< RTSdataSize; i++)
    _RTSdata[i] = dtsm._RTSdata[i];
  _debug_dttrackmatch = dtsm._debug_dttrackmatch;
  // cout << "DTMatch assignment operator done" << endl; 
  return *this;
}






void DTMatch::setMatchingStub(int lay, int phi, int theta, 
			       GlobalVector position, GlobalVector direction) 
{ 
  // Ignazio
  _stub_phi[lay]   = phi;
  _stub_theta[lay] = theta; 
  if( position.mag2() == 0. ) {
    _stub_rho[lay] = NAN;
    _stub_x[lay]   = NAN;
    _stub_y[lay]   = NAN;
    _stub_z[lay]   = NAN;
  }
  else {
    _stub_x[lay] = position.x();
    _stub_y[lay] = position.y();
    _stub_z[lay] = position.z();
    _stub_rho[lay] = position.perp();
    _stub_phiCMS[lay] = stub_phiCMS(lay);
  }
  _stub_position[lay]  = position;
  _stub_direction[lay] = direction;
  _flagMatch[lay] = true; 
  return; 
}





std::string DTMatch::writeMatchingStubs() const {
  // Ignazio
  std::ostringstream outString;
  DTMatchingStubSet::const_iterator st = _matching_stubs.begin();
  for(size_t i = 0; i<_matching_stubs.size(); i++) {
    outString << (*st)->id() << endl;
    ++st;
  }
  return outString.str();
}    





std::string DTMatch::writeMatchingStubs(size_t d) const {
  // Ignazio
  std::ostringstream outString;
  outString << "DTMatch " << d << " total of matchingStubs " 
	    << _matching_stubs.size() << "\n"; 
  DTMatchingStubSet::const_iterator st = _matching_stubs.begin();
  for(size_t i = 0; i<_matching_stubs.size(); i++) {
    outString << (*st)->id() << endl;
    ++st;
  }
  return outString.str();
}    





//===============================================================================
void DTMatch::extrapolateToTrackerLayer(int l) {

  /**
     Argument "l" above is "our" id for tracker layers in use: 6 (six) of them,
     namely 0,1,2,3,8,9.
  **/

  // ***************************************************************************
  // *** Compute DT predicted phi and theta on tracker layers (each wheel)   ***
  // ***************************************************************************
 
  // 090126 SV : compute Predicted in tracker layers from Ext 
  // NB needs to be extrapolated because of bending...
  // Ext = m Bending + q = Bti - Predicted, where Bti = Trigger + sector
  // Predicted = Trigger - (m Bending + q) + sector
  // Predicted for each tracker layer 0...5, from mb1 and mb2 stations.
  
  // PLZ (April 6 2009): Updated for longbarrel October 16 2009
  // extrapolation by wheel and digitization of extrapolated tracks and search 
  // windows.

  // MB1 parameters for tracker layers 0,1,2,3,8,9
  float m1[5][6] = {
  		   {13.83,13.68,13.25,13.10,11.43,11.29},
                   {14.44,14.29,13.85,13.71,12.02,11.88}, 
		   {14.63,14.49,14.05,13.91,12.22,12.08}, 
		   {14.47,14.33,13.89,13.75,12.06,11.92}, 
		   {13.18,13.01,12.65,12.56,11.07,10.94} };
 /* float q1[5][6] = {
                   { 6.94,  6.84,  6.66,  6.56,  5.69,  5.65},
                   { 5.42,  5.35,  5.21,  5.15,  4.46,  4.42}, 
		   { 4.11,  4.07,  3.96,  3.91,  3.47,  3.42}, 
		   { 9.20,  9.09,  8.82,  8.74,  7.66,  7.56},
		   { 8.84,  8.79,  8.46,  8.35,  7.27,  7.19} };*/
  // MB2 parameters for tracker layers 0,1,2,3,8,9
  float m2[5][6] = {
                   {23.01,22.78,22.12,21.90,19.37,19.16}, 
                   {23.87,23.63,23.00,22.78,20.25,20.02}, 
		   {24.10,23.88,23.23,23.01,20.47,20.25}, 
		   {23.87,23.63,23.00,22.78,20.25,20.02}, 
		   {23.01,22.78,22.12,21.90,19.37,19.16} };
  /*float q2[5][6] = {
                   {20.94, 20.68, 20.14, 19.85, 17.48, 17.24},
                   {16.49, 16.36, 15.91, 15.76, 13.89, 13.75},
		   { 9.78,  9.71,  9.44,  9.35,  8.29,  8.23},
		   {18.91, 18.67, 18.17, 17.96, 15.90, 15.79},
		   {20.95, 20.76, 20.16, 19.95, 17.53, 17.39} };*/

  // functions to compute extrapolation windows
  // MB1 parameters for computing errors on phi_ts and phib for each wheel
  float aphi1[5]=  {0.0003225,0.0006993,0.0004521,0.0006889,0.0009605};
  float bphi1[5]=  {0.7658,0.6378,0.6488,0.6704,0.7098};
  float cphi1[5]=  {0.4568,0.7688,0.3860,0.3975,0.9181};
  
  float aphib1[5]=  {0.0006645,0.0004122,0.0005121,0.0005569};
  float bphib1[5]=  {0.08228,0.06362,0.07275,0.05844,0.07803};
  float cphib1[5]=  {2.300,1.302,1.272,1.360,1.352};
  
  // MB2 parameters for computing errors on phi_ts and phib for each wheel  
  float aphi2[5]=  {0.002842,0.005303,0.003674,0.006782,0.005939};
  float bphi2[5]=  {1.177,0.969,1.012,0.9037,1.054};
  float cphi2[5]=  {0.9598,1.244,0.9621,1.626,1.446};  
  
  float aphib2[5]=  {0.001499,0.001262,0.001417,0.001571,0.001566};
  float bphib2[5]=  {0.1115,0.09691,0.09870,0.08293,0.1111};
  float cphib2[5]=  {1.320,1.349,1.360,1.434,1.373};
  
  // 090209 SV : compute theta predicted: do not extrapolate: there is NO bending!
  // errors on theta (layer and wheel dependent, but not pT dependent, 
  // for the moment --> FIX)
  float sigma_theta1[5][6]={
  		   {248.,220.,165.,155., 85., 82.},
                   {439.,392.,292.,269.,142.,137.}, 
		   {590.,523.,387.,341.,189.,176.}, 
		   {437.,371.,280.,254.,143.,132.}, 
		   {256.,228.,174.,158., 88., 85.} };
  float sigma_theta2[5][6]={
  		   {303.,264.,199.,186., 99., 96.},
                   {408.,431.,322.,294.,154.,147.}, 
		   {586.,522.,384.,356.,186.,179.}, 
		   {482.,427.,317.,293.,154.,148.}, 
		   {315.,276.,208.,191.,101., 98.} };
  
  // Integer calculations: shift by 3 bits to preserve resolution
  int bit_shift = 8; 
  
  // station mb1
  if(station()==1 ){//&& flagBxOK() ){
    int iwh 	= wheel()+2;
    int phib1 	= phib_ts();
    int phi_ext = 
      static_cast<int>(m1[iwh][l]*bit_shift)*phib1; 
      //+ static_cast<int>(q1[iwh][l]*bit_shift);
    // PLZ: extrapolation seems worse for code<16, try not to do it in this case 
    // SV 090505 FIXED: extrapolate for code<16 and take 5GeV bending cut
    //if(code() >= 16)
    int phi_mb1_track =	
      phi_ts() - phi_ext/bit_shift + 
      static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
    //else 
    //  phi_mb1_track = phi_ts() 
    //			+ static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
    
    if(phi_mb1_track < 0)
      phi_mb1_track += static_cast<int>(2.*TMath::Pi()*4096.);
    
    // for sigma computation abs value is fine
    phib1 = abs(phib1);
    //	  if(code() < 16) phib1 = 70;		// 9 GeV cut	  
    if(code() < 16) phib1 = 129; 	// 5.5 GeV cut	  
    //	  if(code() < 16) phib1 = 178; 	// 4.5 GeV cut
    
    float sigma_phi  = aphi1[iwh]*phib1*phib1 + bphi1[iwh]*phib1 + cphi1[iwh];
    float sigma_phib  = aphib1[iwh]*phib1*phib1 + bphib1[iwh]*phib1 + cphib1[iwh];
    int sigma_phi_mb1_track = 
      static_cast<int>(sqrt(sigma_phi*sigma_phi+
			    m1[iwh][l]*m1[iwh][l]*sigma_phib*sigma_phib));
    
    int theta_mb1_track= static_cast<int>(thetaCMS()*4096);
    int sigma_theta_mb1_track = static_cast<int>(sigma_theta1[iwh][l]);
    
    if(_debug_dttrackmatch) {
      cout << "Match at bx # " << bx() 
	   << " extrapolate MB1 to layer " << our_to_tracker_lay_Id(l) 
	   << " phiPredicted " << phi_mb1_track << " +- " << sigma_phi_mb1_track 
	   << " from phi Trigger " << phi_ts()
	   << " and phi Bending " << (phib_ts()) 
	   << " thetaPredicted " << theta_mb1_track << " +- " << sigma_theta_mb1_track 
	   << endl;
    }
    // store in DTMatch	
    // int lay = l; 
    // if(l > 7) lay = l-4;
    // Ignazio 091116 adopting tracker_lay_Id_to_our converter
    //int lay = DTMatch::tracker_lay_Id_to_our[l];
    setPredStubPhi(l,phi_mb1_track,sigma_phi_mb1_track);
    setPredStubTheta(l,theta_mb1_track,sigma_theta_mb1_track);
    setPredSigmaPhiB(sigma_phib);
  }// end MB1
  
  // station mb2
  if(station()==2 ){//&& flagBxOK() ){
    int iwh 	 = wheel()+2;
    int phib2 	 = phib_ts();
    int phi_ext = 
      static_cast<int>(m2[iwh][l]*bit_shift)*phib2 ; 
 //     + static_cast<int>(q2[iwh][l]*bit_shift);
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
    //	  if(code() < 16) phib2 = 48;		// 9 GeV cut	  
    if(code() < 16) phib2 = 89; 	// 5.5 GeV cut	  
    //	  if(code() < 16) phib2 = 100; 	// 4.5 GeV cut
    
    float sigma_phi   = aphi2[iwh]*phib2*phib2 + bphi2[iwh]*phib2 + cphi2[iwh];
    float sigma_phib  = aphib2[iwh]*phib2*phib2 + bphib2[iwh]*phib2 + cphib2[iwh];
    int sigma_phi_mb2_track = 
      static_cast<int>(sqrt(sigma_phi*sigma_phi + 
			    m2[iwh][l]*m2[iwh][l]*sigma_phib*sigma_phib));
    
    int theta_mb2_track = static_cast<int>(thetaCMS()*4096);
    int sigma_theta_mb2_track = static_cast<int>(sigma_theta2[iwh][l]);
    
    if(_debug_dttrackmatch) { 
      cout << "Match at bx # " << bx() 
	   << " extrapolate MB2 to layer " << our_to_tracker_lay_Id(l)
	   << " phiPredicted " << phi_mb2_track << " +- " << sigma_phi_mb2_track 
	   << " from phi Trigger " << phi_ts() 
	   << " and phi Bending " << (phib_ts()) 
	   << " thetaPredicted " << theta_mb2_track << " +- " << sigma_theta_mb2_track 
	   << endl;
    }
    // store in DTMatch
    // int lay = l;
    // if(l > 7) lay = l-4;
    // Ignazio 091116 adopting tracker_lay_Id_to_our converter
    //int lay = DTMatch::tracker_lay_Id_to_our[l];	
    setPredStubPhi(l, phi_mb2_track, sigma_phi_mb2_track);
    setPredStubTheta(l, theta_mb2_track, sigma_theta_mb2_track);
    setPredSigmaPhiB(sigma_phib);
  }//end MB2
  
  return;
}



//===============================================================================
void DTMatch::extrapolateToVertex() {
	
		
	// ***************************************************************************
	// *** Compute DT predicted phi and theta at vertex (each wheel)   ***
	// ***************************************************************************
	
	// 090126 SV : compute Predicted in tracker layers from Ext 
	// NB needs to be extrapolated because of bending...
	// Ext = m Bending + q = Bti - Predicted, where Bti = Trigger + sector
	// Predicted = Trigger - (m Bending + q) + sector
	// Predicted for each tracker layer 0...5, from mb1 and mb2 stations.
	
	// PLZ (June 5 2012): Written for longbarrel
	// extrapolation by wheel and digitization of extrapolated tracks and search 
	// windows.
	
	// MB1 parameters for tracker layers 0,1,2,3,8,9
	float m1[5] = {0.003498,0.003082,0.003178,0.003075,0.003493};
	/* float q1[5]= { 0.002128,0.005072,0.003895,0.006074,0.003151};*/
	// MB2 parameters for tracker layers 0,1,2,3,8,9
	float m2[5]= {0.006189,0.005809,0.005104,0.005107,0.006390};
	/*float q2[5] = {-0.005594,-0.0009157,-0.002244,-0.0001975,-0.007733 };*/
	
	// functions to compute extrapolation windows
	// MB1 parameters for computing errors on phi_ts and phib for each wheel
	float aphi1[5]=  {1.12e-06,2.735e-07,3.105e-07,4.165e-07,1.135e-07};
	float bphi1[5]=  {6.489e-05,1.084e-04,1.014e-04,8.921e-05,7.005e-05};
	float cphi1[5]=  {0.001322,0.0006003,0.0005877,0.0009173,0.00135};
	
	float aphib1[5]=  {0.0006645,0.0004122,0.0005121,0.0005569};
	float bphib1[5]=  {0.08228,0.06362,0.07275,0.05844,0.07803};
	float cphib1[5]=  {2.300,1.302,1.272,1.360,1.352};
	
	// MB2 parameters for computing errors on phi_ts and phib for each wheel  
	float aphi2[5]=  {5.894e-06,1.55e-06,2.126e-06,1.997e-06,4.981e-06};
	float bphi2[5]=  {-2.35e-05,1.266e-04,6.119e-05,9.213e-05,3.159e-05};
	float cphi2[5]=  {0.002324,0.0009209,0.001509,0.001235,0.00197};  
	
	float aphib2[5]=  {0.001499,0.001262,0.001417,0.001571,0.001566};
	float bphib2[5]=  {0.1115,0.09691,0.09870,0.08293,0.1111};
	float cphib2[5]=  {1.320,1.349,1.360,1.434,1.373};
 	
	// 090209 SV : compute theta predicted: do not extrapolate: there is NO bending!
	// errors on theta (layer and wheel dependent, but not pT dependent, 
	// for the moment --> FIX)
	float sigma_theta1[5]={248.,439.,590.,437.,256.};
	float sigma_theta2[5]={303.,408.,586.,482.,315.};
	
	
	// station mb1
	if(station()==1 ){//&& flagBxOK() ){
		int iwh 	= wheel()+2;
		int phib1 	= phib_ts();
		int phi_ext = 
		static_cast<int>(m1[iwh]*phib1*4096.); 
		//+ static_cast<int>(q1[iwh][l]*bit_shift);
		// PLZ: extrapolation seems worse for code<16, try not to do it in this case 
		// SV 090505 FIXED: extrapolate for code<16 and take 5GeV bending cut
		//if(code() >= 16)
		int phi_mb1_track =	
		phi_ts() - phi_ext + 
		static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
		//else 
		//  phi_mb1_track = phi_ts() 
		//			+ static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
		
		if(phi_mb1_track < 0)
			phi_mb1_track += static_cast<int>(2.*TMath::Pi()*4096.);
	
		
		// for sigma computation abs value is fine
		phib1 = abs(phib1);
		//	  if(code() < 16) phib1 = 70;		// 9 GeV cut	  
		if(code() < 16) phib1 = 129; 	// 5.5 GeV cut	  
		//	  if(code() < 16) phib1 = 178; 	// 4.5 GeV cut
		
		float sigma_phi  = (aphi1[iwh]*phib1*phib1 + bphi1[iwh]*phib1 + cphi1[iwh])*4096.;
		float sigma_phib  = (aphib1[iwh]*phib1*phib1 + bphib1[iwh]*phib1 + cphib1[iwh])*4096.;
		int sigma_phi_mb1_track = 
		static_cast<int>(sqrt(sigma_phi*sigma_phi+
							  m1[iwh]*m1[iwh]*sigma_phib*sigma_phib));
		
		int theta_mb1_track= static_cast<int>(thetaCMS()*4096);
		int sigma_theta_mb1_track = static_cast<int>(sigma_theta1[iwh]);
		
		if(_debug_dttrackmatch) {
			cout << "Match at bx # " << bx() 
			<< " extrapolate MB1 to vertex " 
			<< " phiPredicted " << phi_mb1_track << " +- " << sigma_phi_mb1_track 
			<< " from phi Trigger " << phi_ts()
			<< " and phi Bending " << (phib_ts()) 
			<< " thetaPredicted " << theta_mb1_track << " +- " << sigma_theta_mb1_track 
			<< endl;
		}
		// store in DTMatch	
		setPredStubPhi(phi_mb1_track,sigma_phi_mb1_track);
		setPredStubTheta(theta_mb1_track,sigma_theta_mb1_track);
//		setPredSigmaPhiB(sigma_phib);
	}// end MB1
	
	// station mb2
	if(station()==2 ){//&& flagBxOK() ){
		int iwh 	 = wheel()+2;
		int phib2 	 = phib_ts();
		int phi_ext = 
		static_cast<int>(m2[iwh]*phib2*4096 ); 
		//     + static_cast<int>(q2[iwh][l]*bit_shift);
		// PLZ: extrapolation seems worse for code<16, try not to do it in this case
		// SV 090505 FIXED: extrapolate for code<16 and take 5GeV bending cut
		//if(code() >= 16)
		int phi_mb2_track = 
		phi_ts() - phi_ext + 
		static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
		//else 
		//	phi_mb2_track = 	phi_ts() 
		//				+ static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
		if(phi_mb2_track < 0)
			phi_mb2_track += static_cast<int>(2.*TMath::Pi()*4096.);
		
		// for sigma computation abs value is fine
		phib2 = abs(phib2);
		//	  if(code() < 16) phib2 = 48;		// 9 GeV cut	  
		if(code() < 16) phib2 = 89; 	// 5.5 GeV cut	  
		//	  if(code() < 16) phib2 = 100; 	// 4.5 GeV cut
		
		float sigma_phi   = (aphi2[iwh]*phib2*phib2 + bphi2[iwh]*phib2 + cphi2[iwh])*4096.;
		float sigma_phib  = (aphib2[iwh]*phib2*phib2 + bphib2[iwh]*phib2 + cphib2[iwh])*4096.;
		int sigma_phi_mb2_track = 
		static_cast<int>(sqrt(sigma_phi*sigma_phi + 
							  m2[iwh]*m2[iwh]*sigma_phib*sigma_phib));
		
		int theta_mb2_track = static_cast<int>(thetaCMS()*4096);
		int sigma_theta_mb2_track = static_cast<int>(sigma_theta2[iwh]);
		
		if(_debug_dttrackmatch) { 
			cout << "Match at bx # " << bx() 
			<< " extrapolate MB2 to vertex " 
			<< " phiPredicted " << phi_mb2_track << " +- " << sigma_phi_mb2_track 
			<< " from phi Trigger " << phi_ts() 
			<< " and phi Bending " << (phib_ts()) 
			<< " thetaPredicted " << theta_mb2_track << " +- " << sigma_theta_mb2_track 
			<< endl;
		}
		// store in DTMatch
		setPredStubPhi(phi_mb2_track, sigma_phi_mb2_track);
		setPredStubTheta(theta_mb2_track, sigma_theta_mb2_track);
//		setPredSigmaPhiB(sigma_phib);
	}//end MB2
	
	return;
}




int DTMatch::corrPhiBend1ToCh2(int phib2) {
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





int DTMatch::corrSigmaPhiBend1ToCh2(int phib2, int sigma_phib2) {
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





void DTMatch::print() {

  cout 	<< "DTMatch : wh " << wheel() << ", st " << station() << ", se " << sector() 
	<< ", bx " << bx() << ", code " << code() << " rejection " << flagReject() 
	<< endl;  
}






//-------------------------------------------------------------------------------

void DTMatch::encoderPt()
// **********************************************************
// *** 100513 PLZ PT priority encoder - Mu_x_y candidates ***
// **********************************************************
// Modifications by Ignazio
{ 
  float nsphi[6] = {10.,10.,10.,10.,10.,10.}; 
  int phimax = static_cast<int>(2*TMath::Pi()*4096.); // rescaling 12-bit encode
  if( flagBxOK() && !flagReject() ) {	
    // compute phi distance between stubs and predicted position as number of 
    // sigmas for each tracker layer; that is nsphi[lay]
    for(int lay = 0; lay<6; lay++) {
      if(isMatched(lay)) {
	int dtphi    = predPhi(lay);
	int sdtphi   = predSigmaPhi(lay);
	int tkphi    = stubPhi(lay); 
	int dist_phi = abs(dtphi-tkphi);	  
	// Reminder: 2pi round window !!!!!
	int dist_phi_max = abs(dtphi+phimax-tkphi);
	int dist_phi_min = abs(dtphi-phimax-tkphi);
	if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
	if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
	nsphi[lay] = static_cast<float>(dist_phi)/static_cast<float>(sdtphi);
      }
    }
    // priority encoding: choose closest hit in layer 2 or 3 and 
    // layer 0 or 1 and computed PT accordingly 
    float inv_Pt = 1000.;
	  
    if(isMatched(3) && isMatched(2)) 
      {
	if( nsphi[3] <= nsphi[2]) 
	  {	     
	    if(isMatched(1) && isMatched(0) && nsphi[0] <= nsphi[1]) 
	      inv_Pt = 1./Pt(string("Mu_3_0"));	     
	    if(isMatched(1) && isMatched(0) && nsphi[0] > nsphi[1]) 
	      inv_Pt = 1./Pt(string("Mu_3_1"));	     
	    if(isMatched(1) && ! isMatched(0))
	      inv_Pt = 1./Pt(string("Mu_3_1"));	     
	    if(!isMatched(1) && isMatched(0)) 
	      inv_Pt = 1./Pt(string("Mu_3_0"));     
	    if(! isMatched(1) && ! isMatched(0))
	      inv_Pt = 1./Pt(string("Mu_3_2"));
	  } 
	else if (nsphi[3] > nsphi[2]) 
	  {     
	    if(isMatched(1) && isMatched(0) && nsphi[0] <= nsphi[1]) 
	      inv_Pt = 1./Pt(string("Mu_2_0"));	     
	    if(isMatched(1) && isMatched(0) && nsphi[0] > nsphi[1]) 
	      inv_Pt = 1./Pt(string("Mu_2_1"));	     
	    if(isMatched(1) && ! isMatched(0))
	      inv_Pt = 1./Pt(string("Mu_2_1"));	 
	    if(! isMatched(1) && isMatched(0))
	      inv_Pt = 1./Pt(string("Mu_2_0"));	
	    if(! isMatched(1) && ! isMatched(0))
	      inv_Pt = 1./Pt(string("Mu_3_2"));      
	  }
      }    
    if(isMatched(3)) 
      {    
	if(isMatched(1) && isMatched(0) && nsphi[0] <= nsphi[1]) 
	  inv_Pt = 1./Pt(string("Mu_3_0"));	     
	if(isMatched(1) && isMatched(0) && nsphi[0] > nsphi[1]) 
	  inv_Pt = 1./Pt(string("Mu_3_1"));	     
	if(isMatched(1) && ! isMatched(0))
	  inv_Pt = 1./Pt(string("Mu_3_1"));	     
	if(! isMatched(1) && isMatched(0))
	  inv_Pt = 1./Pt(string("Mu_3_0"));
      }    
    if(isMatched(2)) 
      {  
	if(isMatched(1) && isMatched(0) && nsphi[0] <= nsphi[1]) 
	  inv_Pt = 1./Pt(string("Mu_2_0"));	     
	if(isMatched(1) && isMatched(0) && nsphi[0] > nsphi[1]) 
	  inv_Pt = 1./Pt(string("Mu_2_1"));	     
	if(isMatched(1) && ! isMatched(0))
	  inv_Pt = 1./Pt(string("Mu_2_1"));	     
	if(! isMatched(1) && isMatched(0))
	  inv_Pt = 1./Pt(string("Mu_2_0"));
      }
    if(!isMatched(3) && !isMatched(2)) 
      {     
	if(isMatched(1) && isMatched(0)) 
	  inv_Pt = 1./Pt(string("Mu_1_0"));
      } 
      
    /*
      if(isMatched(3) && isMatched(0)) inv_Pt = 1./Pt(string("Mu_3_0"));
      else if (isMatched(3) && isMatched(1)) inv_Pt = 1./Pt(string("Mu_3_1"));
      else if (isMatched(2) && isMatched(0)) inv_Pt = 1./Pt(string("Mu_2_0"));
      else if (isMatched(2) && isMatched(1)) inv_Pt = 1./Pt(string("Mu_2_1"));
      else if (isMatched(3) && isMatched(2)) inv_Pt = 1./Pt(string("Mu_3_2"));
      else if (isMatched(1) && isMatched(0)) inv_Pt = 1./Pt(string("Mu_2_0"));
    */
      
    // set selected value
    if (inv_Pt < 1000) setPtEncoder(inv_Pt);     
  } // end if( flagBxOK() && ! flagReject() )
}





void DTMatch::averagePt()
// **********************************************************
// *** 100513 PLZ PT averaging - Mu_x_y candidates *** 
// *** stubs inner layers only ***
// **********************************************************
// Modifications by Ignazio
{ 
// Get Pt estimations of all couples
  if( flagBxOK() && ! flagReject() ) {
    float inv_Pt[6] = {10000.,10000.,10000.,10000.,10000.,10000.};
    if(isMatched(3) && isMatched(2) && Pt(string("Mu_3_2")) < 10000) 
      inv_Pt[0] = 1./Pt(string("Mu_3_2"));
    if(isMatched(3) && isMatched(1) && Pt(string("Mu_3_1")) < 10000) 
      inv_Pt[1] = 1./Pt(string("Mu_3_1"));
    if(isMatched(3) && isMatched(0) && Pt(string("Mu_3_0")) < 10000) 
      inv_Pt[2] = 1./Pt(string("Mu_3_0"));
    if(isMatched(2) && isMatched(1) && Pt(string("Mu_2_1")) < 10000) 
      inv_Pt[3] = 1./Pt(string("Mu_2_1"));
    if(isMatched(2) && isMatched(0) && Pt(string("Mu_2_0")) < 10000) 
      inv_Pt[4] = 1./Pt(string("Mu_2_0"));
    if(isMatched(1) && isMatched(0) && Pt(string("Mu_1_0")) < 10000) 
      inv_Pt[5] = 1./Pt(string("Mu_1_0"));
    float average_inv_Pt = 0.;
    int n_meas = 0;
    for(int i = 0; i < 6; i++){
// Pt cut to participate to bin averaging: 
// temporarily set fixed at 4 GeV (need to insert in configuration later)
    if(inv_Pt[i] < 0.25) {;
         average_inv_Pt = average_inv_Pt +inv_Pt[i];
         n_meas++;
       }
    } 
    if(n_meas > 0) average_inv_Pt =average_inv_Pt/static_cast<float>(n_meas);
    // set selected value
    if (fabs(average_inv_Pt) > 0) setPtAverage(average_inv_Pt);     
  } // end if( flagBxOK() && ! flagReject() )
}


void DTMatch::averagePtTracklet()
// **********************************************************
// *** 100513 PLZ PT averaging - Mu_x_y candidates ***
// *** tracklets full longbarrel ***
// **********************************************************
// Modifications by Ignazio
{ 
	if( flagBxOK() && ! flagReject() ) {	
		// priority encoding: choose closest hit in layer 2 or 3 and 
		// layer 0 or 1 and computed PT accordingly
		float inv_Pt[3] = {10000.,10000.,10000.};
		if( Pt(string("Mu_SL0_SL1")) < 10000) 
			inv_Pt[0] = 1./Pt(string("Mu_SL0_SL1"));
		if( Pt(string("Mu_SL0_SL4")) < 10000) 
			inv_Pt[1] = 1./Pt(string("Mu_SL0_SL4"));
		if( Pt(string("Mu_SL1_SL4")) < 10000) 
			inv_Pt[2] = 1./Pt(string("Mu_SL1_SL4"));
		float average_inv_Pt = 0.;
		int n_meas = 0;
		for(int i = 0; i < 3; i++){
			// Pt cut to participate to bin averaging: 
			// temporarily set fixed at 4 GeV (need to insert in configuration later)
			if(inv_Pt[i] < 0.25) {;
				average_inv_Pt = average_inv_Pt +inv_Pt[i];
				n_meas++;
			}
		} 
		if(n_meas > 0) average_inv_Pt =average_inv_Pt/static_cast<float>(n_meas);
		// set selected value
		if (fabs(average_inv_Pt) > 0) setPtAverageTracklet(average_inv_Pt);
	} // end if( flagBxOK() && ! flagReject() )
}






std::string DTMatch::writePhiStubToPredictedDistance() const {
  // Ignazio
  std::ostringstream outString;   
  float nsphi[6] = {10.,10.,10.,10.,10.,10.};   
  int phimax = static_cast<int>(2*TMath::Pi()*4096.);
  if( flagBxOK() && ! flagReject() ) {	
    // compute phi distance between stubs and predicted position as number of 
    // sigmas for each tracker layer 
    for(int lay = 0; lay<6; lay++) {
      if(isMatched(lay)) {
	int dtphi    = predPhi(lay);
	int sdtphi   = predSigmaPhi(lay);
	int tkphi    = stubPhi(lay); 
	int dist_phi = abs(dtphi-tkphi);	  
	// Reminder: 2pi round window !!!!!
	int dist_phi_max = abs(dtphi+phimax-tkphi);
	int dist_phi_min = abs(dtphi-phimax-tkphi);
	if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
	if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
	nsphi[lay] = static_cast<float>(dist_phi)/static_cast<float>(sdtphi);
      }
    }
  }
  outString << "distance per layer in number of sigmas:\n   " << nsphi[0] 
	    << " "<< nsphi[1] << " "<< nsphi[2] << " "<< nsphi[3] 
	    << " "<< nsphi[4] << " "<< nsphi[5] << endl;
  return outString.str();
}




void DTMatch::assign_encoderPtBin() 
// ***************************************************************************
// *** 100513 PLZ PT bin after priority encoder choice - Mu_x_y candidates ***
// ***************************************************************************
// thresholds (and lower PT values) for MU_x_y pt calculations
//----------------------------------------------------------------------------
// Modifications by Ignazio
{
  float invPT_cut_ST1[21] = { 0.3265, 0.2483, 0.2016, 0.1697, 0.1471, 0.1161, 0.0962, 
			      0.0822, 0.0675, 0.0570, 0.0456, 0.0381, 0.0327, 0.0287, 
			      0.0232, 0.0196, 0.0170, 0.0150, 0.0124, 0.0106, 0.0090 };
  
  float invPT_cut_ST2[21] = { 0.7284, 0.2939, 0.2328, 0.1944, 0.1667, 0.1304, 0.1077, 
			      0.0917, 0.0750, 0.0640, 0.0511, 0.0428, 0.0371, 0.0325, 
			      0.0266, 0.0226, 0.0198, 0.0177, 0.0148, 0.0129, 0.0115 };
  
  float PT_val[22] = { 0., 4., 5., 6., 7., 8., 10., 12., 14., 17., 20., 25., 
		       30., 35., 40., 50., 60., 70., 80., 100., 120., 140.,};
  
  set_encoderPtBin(PT_val[0]);
  if( flagBxOK() && ! flagReject() && flagPt() ) 
    {	
      int done = 0;
      int stat = station();
      float invPt = 1./Pt_encoder();
      for (int i = 0; i < 21 ;i++) {
	if(stat == 1 && invPt > invPT_cut_ST1[i]) 
	  {
	    set_encoderPtBin(PT_val[i]); 
	    done = 1;  
	  }
	if(stat == 2 && invPt > invPT_cut_ST2[i]) 
	  {
	    set_encoderPtBin(PT_val[i]);
	    done = 1; 
	  }
	if (done == 1) break;
      }
      if (done == 0)  set_encoderPtBin(PT_val[21]);
    }
}





void DTMatch::assign_averagePtBin() 
// ***************************************************************************
// *** 100513 PLZ PT bin after averaging - Mu_x_y candidates ***
// ***************************************************************************
// thresholds (and lower PT values) for MU_x_y pt calculations
//----------------------------------------------------------------------------
// Modifications by Ignazio
{
  float invPT_cut_ST1[21] = { 0.3296,	0.2454,	0.1985,	0.1674,	0.1449,	0.1144,	0.0945,
			      0.0808,	0.0663,	0.0561,	0.0448,	0.0373,	0.0320,	0.0283,
			      0.0227,	0.0190,	0.0165,	0.0146,	0.0119,	0.0101,	0.0089};
  
  float invPT_cut_ST2[21] = { 0.3754,	0.2669,	0.2109,	0.1746,	0.1505,	0.1178,	0.0973,
			      0.0829,	0.0679,	0.0575,	0.0460,	0.0387,	0.0333,	0.0294,
			      0.0241,	0.0205,	0.0181,	0.0162,	0.0136,	0.0121,	0.0108 };
  
  float PT_val[22] = { 0., 4., 5., 6., 7., 8., 10., 12., 14., 17., 20., 25., 
		       30., 35., 40., 50., 60., 70., 80., 100., 120., 140.,};
  
  if( flagBxOK() && ! flagReject() && flagPt() ) 
    {	
      int done = 0;
      int stat = station();
      float invPt = 1./Pt_average();
      for (int i = 0; i < 21 ;i++) {
	if(stat == 1 && invPt > invPT_cut_ST1[i]) 
	  {
	    set_averagePtBin(PT_val[i]); 
	    done = 1;  
	  }
	if(stat == 2 && invPt > invPT_cut_ST2[i]) 
	  {
	    set_averagePtBin(PT_val[i]);
	    done = 1; 
	  }
	if (done == 1) break;
      }
      if (done == 0)  set_averagePtBin(PT_val[21]);
    }
}



void DTMatch::assign_averagePtBinTracklet() 
// ***************************************************************************
// *** 100513 PLZ PT bin after averaging - Mu_x_y candidates ***
// ***************************************************************************
// thresholds (and lower PT values) for MU_x_y pt calculations
//----------------------------------------------------------------------------
// Modifications by Ignazio
{
	float invPT_cut_ST1[21] = { 0.3296,	0.2454,	0.1985,	0.1674,	0.1449,	0.1144,	0.0945,
		0.0808,	0.0663,	0.0561,	0.0448,	0.0373,	0.0320,	0.0283,
		0.0227,	0.0190,	0.0165,	0.0146,	0.0119,	0.0101,	0.0089};
	
	float invPT_cut_ST2[21] = { 0.3754,	0.2669,	0.2109,	0.1746,	0.1505,	0.1178,	0.0973,
		0.0829,	0.0679,	0.0575,	0.0460,	0.0387,	0.0333,	0.0294,
		0.0241,	0.0205,	0.0181,	0.0162,	0.0136,	0.0121,	0.0108 };
	
	float PT_val[22] = { 0., 4., 5., 6., 7., 8., 10., 12., 14., 17., 20., 25., 
		30., 35., 40., 50., 60., 70., 80., 100., 120., 140.,};
	
	if( flagBxOK() && ! flagReject() && flagPtTracklet() ) 
    {	
		int done = 0;
		int stat = station();
		float invPt = 1./Pt_average_Tracklet();
		for (int i = 0; i < 21 ;i++) {
			if(stat == 1 && invPt > invPT_cut_ST1[i]) 
			{
				set_averagePtBinTracklet(PT_val[i]); 
				done = 1;  
			}
			if(stat == 2 && invPt > invPT_cut_ST2[i]) 
			{
				set_averagePtBinTracklet(PT_val[i]);
				done = 1; 
			}
			if (done == 1) break;
		}
		if (done == 0)  set_averagePtBinTracklet(PT_val[21]);
    }
}





int DTMatch::assign_ptbin(float invPt, int stat) 
// ***************************************************************************
// *** 100513 PLZ assign PT bin for each determination - Mu_x_y candidates ***
// ***************************************************************************
// thresholds (and lower PT values) for MU_x_y pt calculations
//----------------------------------------------------------------------------
// Modifications by Ignazio
{
  float invPT_cut_ST1[21] = { 0.3296,	0.2454,	0.1985,	0.1674,	0.1449,	0.1144,	0.0945,
			      0.0808,	0.0663,	0.0561,	0.0448,	0.0373,	0.0320,	0.0283,
			      0.0227,	0.0190,	0.0165,	0.0146,	0.0119,	0.0101,	0.0089};
  
  float invPT_cut_ST2[21] = { 0.3754,	0.2669,	0.2109,	0.1746,	0.1505,	0.1178,	0.0973,
			      0.0829,	0.0679,	0.0575,	0.0460,	0.0387,	0.0333,	0.0294,
			      0.0241,	0.0205,	0.0181,	0.0162,	0.0136,	0.0121,	0.0108 };
   
  int ptbin = 1000;
  int done = 0;
      for (int i = 0; i < 21 ;i++) {
	if(stat == 1 && invPt > invPT_cut_ST1[i]){
	    ptbin = i; 
	    done = 1;  
	  }
	if(stat == 2 && invPt > invPT_cut_ST2[i]){
	    ptbin = i;
	    done = 1; 
	  }
	if (done == 1) break;
      }
      if (done == 0)  ptbin =21;
   return ptbin ;
}


int DTMatch::assign_L1track_ptbin(float invPt) 
// ***************************************************************************
// *** 1020619 PLZ assign PT bin for Tracker L1 Tracks                      ***
// ***************************************************************************
// thresholds (and lower PT values) for MU_x_y pt calculations
//----------------------------------------------------------------------------
// Modifications by Ignazio
{
  float invPT_cut[21] = { 0.2535,0.2027,0.1690,	0.1448,	0.1267,	0.1014,
    		   	  0.0845,0.0725,0.0597,	0.0508,	0.0407,	0.0340,	
			  0.0292,0.0255,0.0205,	0.0172,	0.0147,	0.0129,	
			  0.0104,0.0086,0.0074};
  
  float PT_val[22] = { 0., 4., 5., 6., 7., 8., 10., 12., 14., 17., 20., 25., 
		       30., 35., 40., 50., 60., 70., 80., 100., 120., 140.,};
  
  int ptval = PT_val[0];   
  int ptbin = 1000;
  int done = 0;
      for (int i = 0; i < 21 ;i++) {
	if(invPt > invPT_cut[i]){
	    ptbin = i; 
	    done = 1;  
	  }
	if (done == 1) break;
      }
      if (done == 0)  ptbin =21;
      
      ptval =PT_val[ptbin];  
//      cout << " pt " << 1./invPt << " bin " << ptbin << endl;
   return ptval ;
}


/******************************************************************************** 
Full long-barrel (true majority no pT cut)
*/
void DTMatch::assign_majorityPtBinFull()
// **********************************************************
// *** 100513 PLZ PT majority - Mu_x_y candidates ***
// **********************************************************
// Modifications by Ignazio
{ 

  int PT_val[22] = { 0, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 
		       30, 35, 40, 50, 60, 70, 80, 100, 120, 140};
  int majorityPtbin = 0;
  if( flagBxOK() && ! flagReject() ) {	
	  
    float inv_Pt[15] = {1000.,1000.,1000.,1000.,1000.,
		                1000.,1000.,1000.,1000.,1000.,
		                1000.,1000.,1000.,1000.,1000.};
	int ptbin[15] = {1000,1000,1000,1000,1000,
					 1000,1000,1000,1000,1000,
					 1000,1000,1000,1000,1000};
	  
	if(isMatched(5) && isMatched(4)){
		inv_Pt[14] = 1./Pt(string("Mu_9_8"));
		ptbin[14] = assign_ptbin(inv_Pt[14],station());
//       cout << Pt(string("Mu_9_8")) << " " << inv_Pt[14] << " " << ptbin[14] << endl;
	}		  
	if(isMatched(5) && isMatched(3)){
		inv_Pt[13] = 1./Pt(string("Mu_9_3"));
		ptbin[13] = assign_ptbin(inv_Pt[13],station());
//      cout << Pt(string("Mu_9_3")) << " " << inv_Pt[13] << " " << ptbin[13] << endl;
	}	  
	if(isMatched(5) && isMatched(2)){
		inv_Pt[12] = 1./Pt(string("Mu_9_2"));
		ptbin[12] = assign_ptbin(inv_Pt[12],station());
//       cout << Pt(string("Mu_9_2")) << " " << inv_Pt[12] << " " << ptbin[12] << endl;
	}	  
	if(isMatched(5) && isMatched(1)){
		inv_Pt[11] = 1./Pt(string("Mu_9_1"));
		ptbin[11] = assign_ptbin(inv_Pt[11],station());
//       cout << Pt(string("Mu_9_1")) << " " << inv_Pt[11] << " " << ptbin[11] << endl;
	}	  
	if(isMatched(5) && isMatched(0)){
		inv_Pt[10] = 1./Pt(string("Mu_9_0"));
		ptbin[10] = assign_ptbin(inv_Pt[10],station());
  //     cout << Pt(string("Mu_9_0")) << " " << inv_Pt[10] << " " << ptbin[10] << endl;
	}	  	  
	if(isMatched(4) && isMatched(3)){
		inv_Pt[9] = 1./Pt(string("Mu_8_3"));
		ptbin[9] = assign_ptbin(inv_Pt[9],station());
//       cout << Pt(string("Mu_8_3")) << " " << inv_Pt[9] << " " << ptbin[9] << endl;
	}	  
    if(isMatched(4) && isMatched(2)){
		inv_Pt[8] = 1./Pt(string("Mu_8_2"));
		ptbin[8] = assign_ptbin(inv_Pt[8],station());
  //     cout << Pt(string("Mu_8_2")) << " " << inv_Pt[8] << " " << ptbin[8] << endl;
	}	  
	if(isMatched(4) && isMatched(1)){
		inv_Pt[7] = 1./Pt(string("Mu_8_1"));
		ptbin[7] = assign_ptbin(inv_Pt[7],station());
    //   cout << Pt(string("Mu_8_1")) << " " << inv_Pt[7] << " " << ptbin[7] << endl;
	}	  
	if(isMatched(4) && isMatched(0)){
		inv_Pt[6] = 1./Pt(string("Mu_8_0"));
		ptbin[6] = assign_ptbin(inv_Pt[6],station());
      // cout << Pt(string("Mu_8_0")) << " " << inv_Pt[6] << " " << ptbin[6] << endl;
	}	  
    if(isMatched(3) && isMatched(2)){
       inv_Pt[5] = 1./Pt(string("Mu_3_2"));
       ptbin[5] = assign_ptbin(inv_Pt[5],station());
   //    cout << Pt(string("Mu_3_2")) << " " << inv_Pt[5] << " " << ptbin[5] << endl;
       }
    if(isMatched(4) && isMatched(1)){
     inv_Pt[4] = 1./Pt(string("Mu_3_1"));
       ptbin[4] = assign_ptbin(inv_Pt[4],station());
  //     cout << Pt(string("Mu_3_1")) << " " << inv_Pt[4] << " " << ptbin[4] << endl;
     }
    if(isMatched(3) && isMatched(0)) {
    inv_Pt[3] = 1./Pt(string("Mu_3_0"));
       ptbin[3] = assign_ptbin(inv_Pt[3],station());
//       cout << Pt(string("Mu_3_0")) << " " << inv_Pt[3] << " " << ptbin[3] << endl;
    }
    if(isMatched(2) && isMatched(1)) {
    inv_Pt[2] = 1./Pt(string("Mu_2_1"));
       ptbin[2] = assign_ptbin(inv_Pt[2],station());
//       cout << Pt(string("Mu_2_1")) << " " << inv_Pt[2] << " " << ptbin[2] << endl;
    }
    if(isMatched(2) && isMatched(0)) {
    inv_Pt[1] = 1./Pt(string("Mu_2_0"));
       ptbin[1] = assign_ptbin(inv_Pt[1],station());
//       cout << Pt(string("Mu_2_0")) << " " << Pt(string("Mu_8_0")) << " " << inv_Pt[1] << " " << ptbin[1] << endl;
    }
    if(isMatched(1) && isMatched(0)) {
    inv_Pt[0] = 1./Pt(string("Mu_1_0"));
       ptbin[0] = assign_ptbin(inv_Pt[0],station());
 //      cout << Pt(string("Mu_1_0")) << " " << inv_Pt[0] << " " << ptbin[0] << endl;
    }
    
    int pt_counts[22] =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
		       0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for(int i = 0; i < 15; i++){
         if(ptbin[i] <1000) pt_counts[ptbin[i]]++;
       }
        for(int i = 1; i < 22; i++){
//	cout << i << " " << pt_counts[i] << endl;
         if(pt_counts[i] >= pt_counts[majorityPtbin]) majorityPtbin = i;
       } 
//    cout << "maj bin " << PT_val[majorityPtbin] << endl;
    set_majorityPtBinFull(PT_val[majorityPtbin]);     
  } // end if( flagBxOK() && ! flagReject() )
 }




/******************************************************************************** 
Inner long-barrel layers [majority,no pT cut, PT match, no average]
*/
void DTMatch::assign_majorityPtBin()
// **********************************************************
// *** 100513 PLZ PT majority - Mu_x_y candidates ***
// **********************************************************
// Modifications by Ignazio
{ 

  int PT_val[22] = { 0, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 
		     30, 35, 40, 50, 60, 70, 80, 100, 120, 140,};
  int majorityPtbin = 0;
  if( flagBxOK() && ! flagReject() ) {	
    // priority encoding: choose closest hit in layer 2 or 3 and 
    // layer 0 or 1 and compute PT accordingly
    float inv_Pt[6] = {1000.,1000.,1000.,1000.,1000.,1000.};
    int ptbin[6] = {1000,1000,1000,1000,1000,1000};
    //*** PLZ begin	  
//    int PT = DTMatch_PT(station(),wheel(), fabs(static_cast<float>(phib_ts())));
    int PTMin = DTMatch_PTMin(station(),wheel(), fabs(static_cast<float>(phib_ts()))); 
    int PTMax = DTMatch_PTMax(station(),wheel(), fabs(static_cast<float>(phib_ts())));
//     cout << " DTMatch " << PT << " " << PTMin << " " << PTMax << endl;
    //*** PLZ end
	  
    if(isMatched(3) && isMatched(2)){
      inv_Pt[0] = 1./Pt(string("Mu_3_2"));
      //*** PLZ begin
      bool PT_Match = false;
      if(inv_Pt[0] > -10000.) {
	int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[0]);	
	int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[0]);		  
	PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
      }
      if(PT_Match) ptbin[0] = assign_ptbin(inv_Pt[0],station());
//       cout << inv_Pt[0] << " " << ptbin[0] << endl;
      //*** PLZ end
    }
    if(isMatched(3) && isMatched(1)){
      inv_Pt[1] = 1./Pt(string("Mu_3_1"));     
      //*** PLZ begin
      bool PT_Match = false;
      if(inv_Pt[1] > -10000.) {
	int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[1]);	
	int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[1]);		  
	PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
      }
      if(PT_Match) ptbin[1] = assign_ptbin(inv_Pt[1],station());
//       cout << inv_Pt[1] << " " << ptbin[1] << endl;
      //*** PLZ end
    }
    if(isMatched(3) && isMatched(0)) {
      inv_Pt[2] = 1./Pt(string("Mu_3_0"));      
      //*** PLZ begin
      bool PT_Match = false;
      if(inv_Pt[2] >-10000.) {
	int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[2]);	
	int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[2]);		  
	PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
      }
      if(PT_Match) ptbin[2] = assign_ptbin(inv_Pt[2],station());
//       cout << inv_Pt[2] << " " << ptbin[2] << endl;
      //*** PLZ end
    }
    if(isMatched(2) && isMatched(1)) {
      inv_Pt[3] = 1./Pt(string("Mu_2_1"));     
      //PLZ begin
      bool PT_Match = false;
      if(inv_Pt[3] >-10000.) {
	int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[3]);	
	int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[3]);		  
	PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
      }
      if(PT_Match) ptbin[3] = assign_ptbin(inv_Pt[3],station());
      // cout << inv_Pt[3] << " " << ptbin[3] << endl;
      //*** PLZ end
    }
    if(isMatched(2) && isMatched(0)) {
      inv_Pt[4] = 1./Pt(string("Mu_2_0"));     
      //PLZ begin
      bool PT_Match = false;
      if(inv_Pt[4] >-10000.) {
	int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[4]);	
	int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[4]);		  
	PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
      }
      if(PT_Match) ptbin[4] = assign_ptbin(inv_Pt[4],station());
 //      cout << inv_Pt[4] << " " << ptbin[4] << endl;
      //*** PLZ end
    }
    if(isMatched(1) && isMatched(0)) {
      inv_Pt[5] = 1./Pt(string("Mu_1_0"));     
      //PLZ begin
      bool PT_Match = false;
      if(inv_Pt[5] >-10000.) {
	int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[5]);	
	int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[5]);		  
	PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
      }
      if(PT_Match) ptbin[5] = assign_ptbin(inv_Pt[5],station());
 //      cout << inv_Pt[5] << " " << ptbin[5] << endl;
      //*** PLZ end
    }
    
    int pt_counts[22] =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
			   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,};
    for(int i = 0; i < 6; i++){
      if(ptbin[i] <1000) pt_counts[ptbin[i]]++;
    }
    for(int i = 1; i < 22; i++){
      // cout << i << " " << pt_counts[i] << endl;
      if( (pt_counts[i] > 0) && (pt_counts[i] >= pt_counts[majorityPtbin]) ) 
	majorityPtbin = i;
    } 
//     cout << "maj bin " << PT_val[majorityPtbin] << endl;
    set_majorityPtBin(PT_val[majorityPtbin]);     
  } // end if( flagBxOK() && ! flagReject() )
}



/******************************************************************************** 
 Inner long-barrel layers [majority, no pT cut, PT match, average if majority =1]
 */
void DTMatch::assign_mixedmodePtBin()
// **********************************************************
// *** 100513 PLZ PT majority - Mu_x_y candidates ***
// **********************************************************
// Modifications by Ignazio
{ 
	
	int PT_val[22] = { 0, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 
		30, 35, 40, 50, 60, 70, 80, 100, 120, 140,};
	int majorityPtbin = 0;
	if( flagBxOK() && ! flagReject() ) {	
		// priority encoding: choose closest hit in layer 2 or 3 and 
		// layer 0 or 1 and compute PT accordingly
		float inv_Pt[6] = {1000.,1000.,1000.,1000.,1000.,1000.};
		int ptbin[6] = {1000,1000,1000,1000,1000,1000};
		//*** PLZ begin	  
		//    int PT = DTMatch_PT(station(),wheel(), fabs(static_cast<float>(phib_ts())));
		int PTMin = DTMatch_PTMin(station(),wheel(), fabs(static_cast<float>(phib_ts()))); 
		int PTMax = DTMatch_PTMax(station(),wheel(), fabs(static_cast<float>(phib_ts())));
		//     cout << " DTMatch " << PT << " " << PTMin << " " << PTMax << endl;
		//*** PLZ end
		
		if(isMatched(3) && isMatched(2)){
			inv_Pt[0] = 1./Pt(string("Mu_3_2"));
			//*** PLZ begin
			bool PT_Match = false;
			if(inv_Pt[0] > -10000.) {
				int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[0]);	
				int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[0]);		  
				PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
			}
			if(PT_Match) ptbin[0] = assign_ptbin(inv_Pt[0],station());
			//       cout << inv_Pt[0] << " " << ptbin[0] << endl;
			//*** PLZ end
		}
		if(isMatched(3) && isMatched(1)){
			inv_Pt[1] = 1./Pt(string("Mu_3_1"));     
			//*** PLZ begin
			bool PT_Match = false;
			if(inv_Pt[1] > -10000.) {
				int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[1]);	
				int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[1]);		  
				PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
			}
			if(PT_Match) ptbin[1] = assign_ptbin(inv_Pt[1],station());
			//       cout << inv_Pt[1] << " " << ptbin[1] << endl;
			//*** PLZ end
		}
		if(isMatched(3) && isMatched(0)) {
			inv_Pt[2] = 1./Pt(string("Mu_3_0"));      
			//*** PLZ begin
			bool PT_Match = false;
			if(inv_Pt[2] >-10000.) {
				int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[2]);	
				int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[2]);		  
				PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
			}
			if(PT_Match) ptbin[2] = assign_ptbin(inv_Pt[2],station());
			//       cout << inv_Pt[2] << " " << ptbin[2] << endl;
			//*** PLZ end
		}
		if(isMatched(2) && isMatched(1)) {
			inv_Pt[3] = 1./Pt(string("Mu_2_1"));     
			//PLZ begin
			bool PT_Match = false;
			if(inv_Pt[3] >-10000.) {
				int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[3]);	
				int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[3]);		  
				PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
			}
			if(PT_Match) ptbin[3] = assign_ptbin(inv_Pt[3],station());
			// cout << inv_Pt[3] << " " << ptbin[3] << endl;
			//*** PLZ end
		}
		if(isMatched(2) && isMatched(0)) {
			inv_Pt[4] = 1./Pt(string("Mu_2_0"));     
			//PLZ begin
			bool PT_Match = false;
			if(inv_Pt[4] >-10000.) {
				int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[4]);	
				int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[4]);		  
				PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
			}
			if(PT_Match) ptbin[4] = assign_ptbin(inv_Pt[4],station());
			//      cout << inv_Pt[4] << " " << ptbin[4] << endl;
			//*** PLZ end
		}
		if(isMatched(1) && isMatched(0)) {
			inv_Pt[5] = 1./Pt(string("Mu_1_0"));     
			//PLZ begin
			bool PT_Match = false;
			if(inv_Pt[5] >-10000.) {
				int PTMin_1 = DTMatch_PTMin(station(), inv_Pt[5]);	
				int PTMax_1 = DTMatch_PTMax(station(), inv_Pt[5]);		  
				PT_Match = DTStubPTMatch( PTMin, PTMax, PTMin_1, PTMax_1);
			}
			if(PT_Match) ptbin[5] = assign_ptbin(inv_Pt[5],station());
			//      cout << inv_Pt[5] << " " << ptbin[5] << endl;
			//*** PLZ end
		}
		
		int pt_counts[22] =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,};
		for(int i = 0; i < 6; i++){
			if(ptbin[i] <1000) pt_counts[ptbin[i]]++;
		}
		for(int i = 1; i < 22; i++){
//			cout << i << " " << pt_counts[i] << endl;
			if( (pt_counts[i] > 0) && (pt_counts[i] >= pt_counts[majorityPtbin]) ) 
				majorityPtbin = i;
		} 
//		cout << majorityPtbin << endl;
		if (pt_counts[majorityPtbin] == 1){
			int ptsum = 0;
			int ptn   = 0;
			for(int i = 0; i < 6; i++){
				if(ptbin[i] <1000) {
					ptsum=ptsum+ptbin[i];
					ptn++;
				}
			}
			if(ptn > 0) majorityPtbin = static_cast<int>(static_cast<float>(ptsum)/static_cast<float>(ptn));
		}
		
//		     cout << "maj bin " << PT_val[majorityPtbin] << endl;
		set_mixedmodePtBin(PT_val[majorityPtbin]);     
	} // end if( flagBxOK() && ! flagReject() )
}


/******************************************************************************** 
Full long-barrel [ majority,no pT cut, no PT match, no average]
*/
void DTMatch::assign_majorityPtBinTracklet()
// **********************************************************
// *** 100513 PLZ PT majority - Mu_x_y candidates ***
// **********************************************************
// Modifications by Ignazio
{ 

  int PT_val[22] = { 0, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 
		     30, 35, 40, 50, 60, 70, 80, 100, 120, 140,};
  int majorityPtbin = 0;
  if( flagBxOK() && ! flagReject() ) {	
    // priority encoding: choose closest hit in layer 2 or 3 and 
    // layer 0 or 1 and compute PT accordingly
    float inv_Pt[3] = {1000.,1000.,1000.};
    int ptbin[3] = {1000,1000,1000};
    //*** PLZ begin	  
 //   int PT = DTMatch_PT(station(),wheel(), fabs(static_cast<float>(phib_ts())));
  //  int PTMin = DTMatch_PTMin(station(),wheel(), fabs(static_cast<float>(phib_ts()))); 
  //  int PTMax = DTMatch_PTMax(station(),wheel(), fabs(static_cast<float>(phib_ts())));
//     cout << " DTMatch " << PT << " " << PTMin << " " << PTMax << endl;
    //*** PLZ end
	if( Pt(string("Mu_SL0_SL1")) < 10000) {
			inv_Pt[0] = 1./Pt(string("Mu_SL0_SL1"));
			ptbin[0] = assign_ptbin(inv_Pt[0],station());
			}
	if( Pt(string("Mu_SL0_SL4")) < 10000) {
			inv_Pt[1] = 1./Pt(string("Mu_SL0_SL4"));
			ptbin[1] = assign_ptbin(inv_Pt[1],station());
			}
	if( Pt(string("Mu_SL1_SL4")) < 10000) {
			inv_Pt[2] = 1./Pt(string("Mu_SL1_SL4"));
			ptbin[2] = assign_ptbin(inv_Pt[2],station());
			}

    int pt_counts[22] =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
			   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,};
    for(int i = 0; i < 3; i++){
      if(ptbin[i] <1000) pt_counts[ptbin[i]]++;
    }
    for(int i = 1; i < 22; i++){
      if( (pt_counts[i] > 0) && (pt_counts[i] >= pt_counts[majorityPtbin]) ) 
	majorityPtbin = i;
    } 
    set_majorityPtBinTracklet(PT_val[majorityPtbin]);     
  } // end if( flagBxOK() && ! flagReject() )
}


/******************************************************************************** 
 Full long-barrel [ majority, pT cut, no PT match, average if = 1]
 */
void DTMatch::assign_mixedmodePtBinTracklet()
// **********************************************************
// *** 100513 PLZ PT majority - Mu_x_y candidates ***
// **********************************************************
// Modifications by Ignazio
{ 
	
	int PT_val[22] = { 0, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 
		30, 35, 40, 50, 60, 70, 80, 100, 120, 140,};
	int majorityPtbin = 0;
	if( flagBxOK() && ! flagReject() ) {	
		// priority encoding: choose closest hit in layer 2 or 3 and 
		// layer 0 or 1 and compute PT accordingly
		float inv_Pt[3] = {1000.,1000.,1000.};
		int ptbin[3] = {1000,1000,1000};
		//*** PLZ begin	  
		//   int PT = DTMatch_PT(station(),wheel(), fabs(static_cast<float>(phib_ts())));
		//  int PTMin = DTMatch_PTMin(station(),wheel(), fabs(static_cast<float>(phib_ts()))); 
		//  int PTMax = DTMatch_PTMax(station(),wheel(), fabs(static_cast<float>(phib_ts())));
		//     cout << " DTMatch " << PT << " " << PTMin << " " << PTMax << endl;
		//*** PLZ end
		if( Pt(string("Mu_SL0_SL1")) < 10000) {
			inv_Pt[0] = 1./Pt(string("Mu_SL0_SL1"));
			if(inv_Pt[0] < 0.25) ptbin[0] = assign_ptbin(inv_Pt[0],station());
		}
		if( Pt(string("Mu_SL0_SL4")) < 10000) {
			inv_Pt[1] = 1./Pt(string("Mu_SL0_SL4"));
			if(inv_Pt[1] < 0.25) ptbin[1] = assign_ptbin(inv_Pt[1],station());
		}
		if( Pt(string("Mu_SL1_SL4")) < 10000) {
			inv_Pt[2] = 1./Pt(string("Mu_SL1_SL4"));
			if(inv_Pt[2] < 0.25) ptbin[2] = assign_ptbin(inv_Pt[2],station());
		}
		
//			   cout << ptbin[0] << " " << ptbin[0] << " " << ptbin[0] << endl;
			   
		int pt_counts[22] =  { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0,};
		for(int i = 0; i < 3; i++){
			if(ptbin[i] <1000) pt_counts[ptbin[i]]++;
		}
		for(int i = 1; i < 22; i++){
			if( (pt_counts[i] > 0) && (pt_counts[i] >= pt_counts[majorityPtbin]) ) 
				majorityPtbin = i;
		} 
		
		if (pt_counts[majorityPtbin] == 1){
			int ptsum = 0;
			int ptn   = 0;
			for(int i = 0; i < 3; i++){
				if(ptbin[i] <1000) {
					ptsum=ptsum+ptbin[i];
					ptn++;
				}
			}
			if(ptn > 0) majorityPtbin = static_cast<int>(static_cast<float>(ptsum)/static_cast<float>(ptn));
		}
		
		set_mixedmodePtBinTracklet(PT_val[majorityPtbin]);     
	} // end if( flagBxOK() && ! flagReject() )
}



int  DTMatch::DTMatch_PT(int station,int wheel, float phib)
{
  int PT =1000;
  if(phib > 0. ){
    /*
      if(station == 1 & abs(wheel) == 2)  
      PT = static_cast<int>(-12.56*exp(-0.05921*phib)+720/phib);
      if(station == 1 & abs(wheel) == 1)  
      PT = static_cast<int>(-9.463*exp(-0.06816*phib)+764.4/phib);
      if(station == 1 & abs(wheel) == 0)  
      PT = static_cast<int>(0.1715*exp(0.01078*phib)+720.5/phib);
      if(station == 2 & abs(wheel) == 2)  
      PT = static_cast<int>(-33.09*exp(-0.62*phib)+437.6/phib);
      if(station == 2 & abs(wheel) == 1)  
      PT = static_cast<int>(1.631*exp(-0.008652*phib)+486.3/phib);
      if(station == 2 & abs(wheel) == 0)  
      PT = static_cast<int>(-38.*exp(-0.407*phib)+528.6/phib);
      cout <<" PT " <<  station << " " << wheel << " " << phib << " " << PT << endl;
    */
    if((station == 1) && (abs(wheel) == 2))  
      PT = static_cast<int>(-13.93*exp(-0.05828*phib)+728.6/phib);
    if((station == 1) && (abs(wheel) == 1))  
      PT = static_cast<int>(-10.34*exp(-0.06771*phib)+768.6/phib);
    if((station == 1) && (abs(wheel) == 0))  
      PT = static_cast<int>(0.1715*exp(0.01078*phib)+720.5/phib);
    if((station == 2) && (abs(wheel) == 2))  
      PT = static_cast<int>(1.749*exp(-0.01927*phib)+419.9/phib);
    if((station == 2) && (abs(wheel) == 1))  
      PT = static_cast<int>(1.3*exp(-0.005566*phib)+488.6/phib);
    if((station == 2) && (abs(wheel) == 0))  
      PT = static_cast<int>(-38.*exp(-0.407*phib)+528.6/phib);
    // cout <<" PT " <<  station << " " << wheel << " " 
    //      << phib << " " << PT << endl;  
  }  
  /*
    int PT_val[22] = { 0, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 
    30, 35, 40, 50, 60, 70, 80, 100, 120, 140,};
    int PT_bin = -1;
    for(int i = 1; i<23; i++) { 
    if(PT < PT_val[i-1]) PT_bin = i-1;
    if(PT < PT_val[i-1]) break;
    }
    if(PT_bin == -1) PT_bin = 21;	       
    return PT_bin;
  */  
  return PT;
}





int  DTMatch::DTMatch_PTMax(int station,int wheel, float phib)
{
  int phib_min = 0;
  //   int PT_bin = 21;
  int PTMax =1000;
  if(phib > 0. ){
    // 2 sigma cut   
    /*
      if(station == 1 & abs(wheel) == 2)  
      phib_min =static_cast<int>((-0.001749*phib*phib+0.9308*phib-3.186)-0.5);
      if(station == 1 & abs(wheel) == 1)  
      phib_min =static_cast<int>((-0.0003879*phib*phib+0.8673*phib-2.326)-0.5);
      if(station == 1 & abs(wheel) == 0)  
      phib_min =static_cast<int>((-0.0003745*phib*phib+0.8441*phib-2.203)-0.5);
      if(station == 2 & abs(wheel) == 2)  
      phib_min =static_cast<int>((-0.00378*phib*phib+0.851*phib-3.062)-0.5);
      if(station == 2 & abs(wheel) == 1)  
      phib_min =static_cast<int>((-0.001702*phib*phib+0.8332*phib-2.778)-0.5);
      if(station == 2 & abs(wheel) == 0)  
      phib_min =static_cast<int>((-0.002871*phib*phib+0.9051*phib-3.278)-0.5);
    */
    // 3 sigma cut   
    if((station == 1) && (abs(wheel) == 2))  
      phib_min =static_cast<int>((-0.001251*phib*phib+0.7838*phib-3.458)-0.5);
    if((station == 1) && (abs(wheel) == 1))  
      phib_min =static_cast<int>((-0.0005819*phib*phib+0.8511*phib-8.269)-0.5);
    if((station == 1) && (abs(wheel) == 0))  
      phib_min =static_cast<int>((-0.000544*phib*phib+0.7625*phib-3.167)-0.5);
    if((station == 2) && (abs(wheel) == 2))  
      phib_min =static_cast<int>((-0.003519*phib*phib+0.6528*phib-3.437)-0.5);
    if((station == 2) && (abs(wheel) == 1))  
      phib_min =static_cast<int>((-0.001781*phib*phib+0.6869*phib-3.344)-0.5);
    if(station == 2 && abs(wheel) == 0)  
      phib_min =static_cast<int>((-0.002934*phib*phib+0.7467*phib-3.621)-0.5);
  }   
  if(phib_min > 0 ) 
    PTMax = DTMatch_PT(station,wheel, static_cast<float>(phib_min));
  //	cout <<" PTMAX " <<  station << " " << wheel << " " 
  //         << phib << " " << phib_min << " " << PTMax << endl; 
  return PTMax;
}





int  DTMatch::DTMatch_PTMin(int station,int wheel, float phib)
{
  int phib_max = 0;
  //   int PT_bin = 21;
  int PTMin = 0;
  if(phib > 0. ){
    // 2 sigma cut
    /*
      if(station == 1 & abs(wheel) == 2)  
      phib_max =static_cast<int>((0.003333*phib*phib+0.9123*phib+5.187)+0.5);
      if(station == 1 & abs(wheel) == 1)  
      phib_max =static_cast<int>((0.0003879*phib*phib+1.133*phib+2.326)+0.5);
      if(station == 1 & abs(wheel) == 0)  
      phib_max =static_cast<int>((0.0003745*phib*phib+1.156*phib+2.203)+0.5);
      if(station == 2 & abs(wheel) == 2)  
      phib_max =static_cast<int>((0.003771*phib*phib+1.15*phib+3.042)+0.5);
      if(station == 2 & abs(wheel) == 1)  
      phib_max =static_cast<int>((0.001719*phib*phib+1.165*phib+2.835)+0.5);
      if(station == 2 & abs(wheel) == 0)  
      phib_max =static_cast<int>((0.002871*phib*phib+1.095*phib+3.278)+0.5);
    */
    // 3 sigma cut
    if((station == 1) && (abs(wheel) == 2))  
      phib_max =static_cast<int>((0.005109*phib*phib+0.8511*phib+8.269)+0.5);
    if((station == 1) && (abs(wheel) == 1))  
      phib_max =static_cast<int>((0.0005819*phib*phib+1.199*phib+3.488)+0.5);
    if((station == 1) && (abs(wheel) == 0))  
      phib_max =static_cast<int>((0.0005618*phib*phib+1.234*phib+3.305)+0.5);
    if((station == 2) && (abs(wheel) == 2))  
      phib_max =static_cast<int>((0.005903*phib*phib+1.206*phib+4.74)+0.5);
    if((station == 2) && (abs(wheel) == 1))  
      phib_max =static_cast<int>((0.00264*phib*phib+1.241*phib+4.317)+0.5);
    if((station == 2) && (abs(wheel) == 0))  
      phib_max =static_cast<int>((0.003406*phib*phib+1.142*phib+4.918)+0.5);
  }
  if(phib_max > 0 ) 
    PTMin = DTMatch_PT(station,wheel, static_cast<float>(phib_max));  
  //	cout <<" PTMin " <<  station << " " << wheel << " " 
  //         << phib << " " << phib_max << " " << PTMin << endl;  
  return PTMin;
}





int  DTMatch::DTMatch_PTMax(int station, float invPT)
{
  float invPT_min = 0;
  int PTMax = 1000 ;
  //   int PT_bin = 21 ;
  // 2 sigma cut
  /*
    if(station == 1)  invPT_min = -0.2275*invPT*invPT+0.9177*invPT-0.0013; 
    if(station == 2)  invPT_min = -0.4917*invPT*invPT+0.9245*invPT-0.0026; 
  */
  // 3 sigma cut
  if(station == 1)  
    invPT_min = 0.2876*invPT*invPT+0.8524*invPT-0.001457; 
  if(station == 2)  
    invPT_min = 0.2486*invPT*invPT+0.9371*invPT-0.002891;   
  PTMax = static_cast<int>(1./invPT_min+0.5);    
  //  if(invPT_min > 0 ) PT_bin = assign_ptbin(invPT_min, station) ;
  //   cout <<" PTMAX " <<  station << " " << 1/invPT << " " 
  //        << 1/invPT_min << " " << PT_bin << endl;
  
  return PTMax;
}





int  DTMatch::DTMatch_PTMin(int station,float invPT)
{
  float invPT_max = 0;
  int PTMin = 0;
  //   int PT_bin = 21 ;
  // 2 sigma cut
  /*
    if(station == 1)  invPT_max = 0.2272*invPT*invPT+1.082*invPT+0.0013;
    if(station == 2)  invPT_max = 0.5482*invPT*invPT+1.068*invPT+0.0027;  
  */
  // 3 sigma cut
  if(station == 1)  invPT_max = 0.9664*invPT*invPT+1.1*invPT+0.00243;
  if(station == 2)  invPT_max = 2.024*invPT*invPT+1.124*invPT+0.005521;  
    PTMin = static_cast<int>(1./invPT_max-0.5);
  //  if(invPT_max > 0 ) PT_bin = assign_ptbin(invPT_max, station) ;
  //   cout <<" PTMin " <<  station << " " << 1/invPT << " " 
  //        << 1/invPT_max << " " << PT_bin << endl;
  
  return PTMin;
}





bool  DTMatch::DTStubPTMatch(int DTPTMin,int DTPTMax,int TKPTMin,int TKPTMax)
{
  bool match = false;
  int PTMin_Match = DTPTMin;
  if(TKPTMin > DTPTMin) PTMin_Match = TKPTMin;
  int PTMax_Match = DTPTMax;
  if(TKPTMax < DTPTMax) PTMax_Match = TKPTMax;
  if(PTMax_Match > PTMin_Match) match = true;
  return match;
}




