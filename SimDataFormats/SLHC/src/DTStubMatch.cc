#ifdef SLHC_DT_TRK_DFENABLE
#include <math.h>

#include "SimDataFormats/SLHC/interface/DTStubMatch.h"

using namespace std;

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
      phi_ts() - phi_ext/bit_shift + static_cast<int>((sector()-1)*TMath::Pi()/6.*4096.);
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
      static_cast<int>(sqrt(sigma_phi*sigma_phi+m1[iwh][l]*m1[iwh][l]*sigma_phib*sigma_phib));

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

bool DTStubMatchSortPredicate(const DTStubMatch* d1, const DTStubMatch* d2)
{
  return (d1->trig_order() < d2->trig_order());
}


//-------------------------------------------------------------------------------

ostream& operator <<(ostream &os, const DTStubMatch &obj)
{
  for(size_t i=0; i<(RTSdataSize-1); i++)
    os << obj.RTSdata(i) << "  ";
  os << obj.RTSdata((RTSdataSize-1)) << endl;
  return os;
}


//------------------------------------------------------------------------------
void DTStubMatch::init()
{
  _GunFiredSingleMuPt = 0.;
  _Pt_value = NAN;    // Ignazio
  _Pt_bin = NAN;      // Ignazio
  _trig_order = -555555555;
  _pred_theta = -555555555;
  _pred_sigma_phib = NAN;
  _delta_theta = -555555555;

  for(int l=0; l<StackedLayersInUseTotal; l++){
    _pred_phi[l] = -555555555;
    _pred_sigma_phi[l] = -555555555;
    _pred_sigma_theta[l] = -555555555;
  }

  _matching_stubs = StubTracklet();
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




DTStubMatch::DTStubMatch() {
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
DTStubMatch::DTStubMatch(int wheel, int station, int sector,
			 int bx, int code, int phi, int phib, float theta, bool flagBxOK,
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
DTStubMatch::DTStubMatch(int wheel, int station, int sector,
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
DTStubMatch::DTStubMatch(const DTStubMatch& dtsm): DTStubMatchPtAlgorithms(dtsm)
{
  // cout << "DTStubMatch copy constructor called" << endl;
  _phi_ts = dtsm.phi_ts();
  _phib_ts = dtsm.phib_ts();
  _flagBxOK = dtsm.flagBxOK();
  _trig_order = dtsm.trig_order();
  _flag_reject = dtsm.flagReject();
  _flagPt = dtsm.flagPt();
  _flag_theta = dtsm.flagTheta();
  _delta_theta = dtsm.deltaTheta();
  _Pt_value = dtsm.Pt_value();
  _Pt_bin = dtsm.Pt_bin();

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
  // cout << "DTStubMatch copy constructor done" << endl;
}



// assignment operator
DTStubMatch& DTStubMatch::operator =(const DTStubMatch& dtsm)
{
  // cout << "DTStubMatch assignment operator called" << endl;
  if (this == &dtsm)      // Same object?
    return *this;         // Yes, so skip assignment, and just return *this.
  this->DTStubMatchPtAlgorithms::operator=(dtsm);
  _phi_ts = dtsm.phi_ts();
  _phib_ts = dtsm.phib_ts();
  _flagBxOK = dtsm.flagBxOK();
  _trig_order = dtsm.trig_order();
  _flag_reject = dtsm.flagReject();
  _flagPt = dtsm.flagPt();
  _flag_theta = dtsm.flagTheta();
  _delta_theta = dtsm.deltaTheta();
  _Pt_value = dtsm.Pt_value();
  _Pt_bin = dtsm.Pt_bin();

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
  // cout << "DTStubMatch assignment operator done" << endl;
  return *this;
}



void DTStubMatch::setMatchStub(int lay, int phi, int theta,
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




//-------------------------------------------------------------------------------

void DTStubMatch::choosePtValue()
// **********************************************************
// *** 100513 PLZ PT priority encoder - Mu_x_y candidates ***
// **********************************************************
// Modifications by Ignazio
{
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
	    if(isMatched(1) && ! isMatched(0))
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
	    if(! isMatched(1) &&  isMatched(0))
	      inv_Pt = 1./Pt(string("Mu_2_1"));
	    if(isMatched(1) && ! isMatched(0))
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
    // set selected value
    if (inv_Pt < 1000) setPtValue(inv_Pt);
  } // end if( flagBxOK() && ! flagReject() )
}




std::string DTStubMatch::writePhiStubToPredictedDistance() const {
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




void DTStubMatch::assignPtBin()
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

  if( flagBxOK() && ! flagReject() && flagPt() )
    {
      int done = 0;
      int stat = station();
      float invPt = 1./Pt_value();
      for (int i = 0; i < 21 ;i++) {
	if(stat == 1 && invPt > invPT_cut_ST1[i])
	  {
	    setPtBin(PT_val[i]);
	    done = 1;
	  }
	if(stat == 2 && invPt > invPT_cut_ST2[i])
	  {
	    setPtBin(PT_val[i]);
	    done = 1;
	  }
	if (done == 1) break;
      }
      if (done == 0)  setPtBin(PT_val[21]);
    }
}

#endif
