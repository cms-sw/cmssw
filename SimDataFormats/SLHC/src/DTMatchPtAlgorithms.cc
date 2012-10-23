#include <math.h>
#include <string>

#include "SimDataFormats/SLHC/interface/DTMatchPtAlgorithms.h"


using namespace std;

// --------------------------------------------------
int const our_to_tracker_lay_Id(int const l) {
  return StackedLayersInUse[l];
}

int const tracker_lay_Id_to_our(int const l) {
  int our = l;
  if(l == 8) our = 4;
  else if(l == 9) our = 5;
  return our;
}
// --------------------------------------------------
int const our_to_tracker_superlay_Id(int const l) {
  return StackedLayersInUse[l];
}

int const tracker_superlay_Id_to_our(int const l) {
  int our = l;
  if(l == 4) our = 2;
  return our;
}
// --------------------------------------------------


static const double Ri_Coil = 315.0;
const double DTMatchPtAlgorithms::_RiCoil = Ri_Coil;
static const double Ro_Coil = 340.0;
const double DTMatchPtAlgorithms::_RoCoil = Ro_Coil;
const double DTMatchPtAlgorithms::_rtilde = 
  (Ri_Coil*Ri_Coil + Ri_Coil*Ro_Coil + Ro_Coil*Ro_Coil)/(3.*Ro_Coil);



// constructor
DTMatchPtAlgorithms::DTMatchPtAlgorithms() {

  _wheel     = -9999999;
  _station   = -9999999;
  _sector    = -9999999;
  _bx        = -9999999;
  _code      = -9999999;

  _matching_stubs_No = 0; 

  _rho       = NAN;
  _phiCMS    = NAN;
  _bendingDT = NAN;
  _thetaCMS  = NAN;
  _eta       = NAN;
  _phiR = _PhiR = _PhiRI = NAN; 
  _deltaPhiR = NAN;    
  _vstub_rho = _vstub_phiCMS = NAN;
  _sqrtDscrm = NAN;
  _alphaDT = NAN;

  for(int l=0; l<StackedLayersInUseTotal; l++){	
    _stub_phi[l] = -555555555;
    _stub_theta[l] = -555555555;
    _stub_x[l] = NAN;
    _stub_y[l] = NAN;
    _stub_z[l] = NAN;
    _stub_rho[l]= NAN;
    _stub_phiCMS[l] = NAN;
    _stub_dephi[l] = NAN;
   for(int j=0; j<l; j++) {
     int idx = (l*(l-1))/2 + j;
     _stubstub_dephi[idx] = NAN;
   }

    _stub_thetaCMS[l] = NAN;
    _stub_position[l] = GlobalVector();
    _stub_direction[l] = GlobalVector();
    _flagMatch[l] = false;
  }
  _MatchingTkTracklets = vector<TrackerTracklet*>(TkSLinUseTotal);
}
 



// copy constructor
DTMatchPtAlgorithms::DTMatchPtAlgorithms(const DTMatchPtAlgorithms& pts): 
  DTMatchPtVariety(pts) 
{
  _wheel = pts._wheel;
  _station = pts._station; 
  _sector = pts._sector; 
  _bx = pts._bx;
  _code = pts._code;
  _phiCMS    = pts._phiCMS;
  _bendingDT = pts._bendingDT; 
  _thetaCMS  = pts._thetaCMS;
  _eta       = pts._eta;
  _rho       = pts._rho;

  for(int l=0; l<StackedLayersInUseTotal; l++){	
    _stub_phi[l] = pts._stub_phi[l];
    _stub_theta[l] = pts._stub_theta[l];
    _flagMatch[l] = pts._flagMatch[l];
    _stub_x[l] = pts._stub_x[l];
    _stub_y[l] = pts. _stub_y[l];
    _stub_z[l] = pts._stub_z[l];
    _stub_rho[l] = pts._stub_rho[l];
    _stub_phiCMS[l] = pts._stub_phiCMS[l];
    _stub_dephi[l] = pts._stub_dephi[l];
    _stub_thetaCMS[l] = pts. _stub_thetaCMS[l];
    _stub_position[l] = pts._stub_position[l];
    _stub_direction[l] = pts. _stub_direction[l];
  }
  for(int j=0;j<STUBSTUBCOUPLES; j++) 
    _stubstub_dephi[j] = pts._stubstub_dephi[j];

  _position = pts.position();
  _direction = pts.direction();

  _alphaDT = pts.alphaDT();
  _sqrtDscrm = pts.sqrtDiscrim();
  _vstub_rho = pts._vstub_rho; 
  _vstub_phiCMS = pts._vstub_phiCMS;

  _phiR = pts._phiR;
  _PhiR = pts._PhiR;
  _PhiRI = pts._PhiRI;
  _deltaPhiR = pts._deltaPhiR;
}



// assignment constructor
  DTMatchPtAlgorithms& 
  DTMatchPtAlgorithms::operator =(const DTMatchPtAlgorithms& pts) {
  if (this == &pts)      // Same object?
    return *this;        // Yes, so skip assignment, and just return *this.
  this->DTMatchPtVariety::operator=(pts);
  _wheel = pts._wheel;
  _station = pts._station; 
  _sector = pts._sector; 
  _bx = pts._bx;
  _code = pts._code;
  _phiCMS    = pts._phiCMS;
  _bendingDT = pts._bendingDT; 
  _thetaCMS  = pts._thetaCMS;
  _eta       = pts._eta;
  _rho       = pts._rho;
  for(int l=0; l<StackedLayersInUseTotal; l++){	
    _stub_phi[l] = pts._stub_phi[l];
    _stub_theta[l] = pts._stub_theta[l];
    _flagMatch[l] = pts._flagMatch[l];
    _stub_x[l] = pts._stub_x[l];
    _stub_y[l] = pts. _stub_y[l];
    _stub_z[l] = pts._stub_z[l];
    _stub_rho[l] = pts._stub_rho[l];
    _stub_phiCMS[l] = pts._stub_phiCMS[l];
    _stub_dephi[l] = pts._stub_dephi[l];
    _stub_thetaCMS[l] = pts. _stub_thetaCMS[l];
    _stub_position[l] = pts._stub_position[l];
    _stub_direction[l] = pts. _stub_direction[l];
  }
  for(int j=0;j<STUBSTUBCOUPLES; j++) 
    _stubstub_dephi[j] = pts._stubstub_dephi[j];

  _position = pts.position();
  _direction = pts.direction();

  _alphaDT = pts.alphaDT();
  _sqrtDscrm = pts.sqrtDiscrim();
  _vstub_rho = pts._vstub_rho; 
  _vstub_phiCMS = pts._vstub_phiCMS;

  _phiR = pts._phiR;
  _PhiR = pts._PhiR;
  _PhiRI = pts._PhiRI;
  _deltaPhiR = pts._deltaPhiR;
  return *this;
}




//-------------------------------------------------------------------------------
float DTMatchPtAlgorithms::phiCMS(const GlobalVector& P) const { 
  float phiCMS = P.phi();
  if(phiCMS < 0.)
    phiCMS += 2. * TMath::Pi();
  if(phiCMS > 2*TMath::Pi())
    phiCMS -= 2 * TMath::Pi();
  return phiCMS;
}



void DTMatchPtAlgorithms::set_stub_dephi(int lay, float dephi) { 	
  _stub_dephi[lay] = dephi; 
  return; 
} 

 

float DTMatchPtAlgorithms::stub_phiCMS(size_t lay) const { 
  float phiCMS = static_cast<float>(_stub_phi[lay])/4096.;
  if(phiCMS < 0.)
    phiCMS += 2. * TMath::Pi();
  if(phiCMS > 2*TMath::Pi())
    phiCMS -= 2 * TMath::Pi();
  return phiCMS;
}



void DTMatchPtAlgorithms::set_stubstub_dephi(int L1, int L2, float dephi) { 
  // call tracker_lay_Id_to_our to be safe
  L1 = tracker_lay_Id_to_our(L1);
  L2 = tracker_lay_Id_to_our(L2);	
  if(L1 > L2) {
    int idx = (L1*(L1-1))/2 + L2;
    _stubstub_dephi[idx] = dephi;
  }
  else if(L2 > L1) {
    int idx = (L2*(L2-1))/2 + L1;
    _stubstub_dephi[idx] = -dephi;
  } 
  return; 
} 



float DTMatchPtAlgorithms::stubstub_dephi(int L1, int L2) const {
  // call tracker_lay_Id_to_our to be safe
  L1 = tracker_lay_Id_to_our(L1);
  L2 = tracker_lay_Id_to_our(L2);
  if(L1 > L2) {
    int idx = (L1*(L1-1))/2 + L2;
    /*
      cout << "(" << L1 << ", " << L2 << ") --> idx = " << idx 
      << ": _stubstub_dephi = " <<  _stubstub_dephi[idx] << endl;
    */
    return _stubstub_dephi[idx];
  }
  else if(L2 > L1) {
    int idx = (L2*(L2-1))/2 + L1;
    /*
      cout << "(" << L1 << ", " << L2 << ") --> idx = " << idx 
      << ": _stubstub_dephi = " <<  _stubstub_dephi[idx] << endl;
    */
    return (- _stubstub_dephi[idx]);
  } 
  else return 0.;
}



//-------------------------------------------------------------------------------
void DTMatchPtAlgorithms::setPt(const edm::ParameterSet&  pSet) 
{
  // Prepare.
  // First define the direction of assumed straight line trajectory of the muon
  // inside DT chambers: 
  double st2_corr = pSet.getUntrackedParameter<double>("station2_correction", 1.0);
  double delta = static_cast<double>(_bendingDT); 
  delta = (_station == 1)? delta: st2_corr*delta;
  _alphaDT = _phiCMS + delta;
  if(_alphaDT <= 0.)
    _alphaDT += 2 * TMath::Pi();
  if(_alphaDT > 2 * TMath::Pi())
    _alphaDT -= 2 * TMath::Pi();
  float phits = (_phiCMS-(_sector-1)*TMath::Pi()/6.)*4096;
  // Then get intercept of that straight line trajectory to the ideal cylindrical 
  // boundary surface of the CMS magnetic field: this will be used as a point
  // belonging to the muon trajectory inside the magnetic field toghether with
  // already matched tracker stubs.
  // The idea id trying to avoid use of the outermost tracker layers (8th and 9nt),
  // stubs, however keeping comparable precision.
  
  // first approach ------------------------------------------------------------ 
  double rhosq = _position.x()*_position.x() + _position.y()*_position.y();
  double rho   = sqrt(rhosq);
  //  double rho = 0. ;
  //  if(_station == 1) rho = 0.00001331*phits*phits+0.00002972*phits+431.1;
  //  if(_station == 2) rho = 0.00001568*phits*phits+0.00001665*phits+512.4;
  _rho = static_cast<float>(rho);
  // cout << phits << " " << _rho << endl;
  double Dscrm  = 1. - rho*rho*sin(delta)*sin(delta)/(_rtilde*_rtilde);
  if( Dscrm < 0.)  
    return;
  _sqrtDscrm = static_cast<float>(sqrt(Dscrm));
  // cout  << _sqrtDscrm << endl;
  float _xerre = _rtilde*cos(_alphaDT)*_sqrtDscrm + rho*sin(delta)*sin(_alphaDT);
  float _yerre = _rtilde*sin(_alphaDT)*_sqrtDscrm - rho*sin(delta)*cos(_alphaDT);
  _phiR = (_yerre>0.)? acos(_xerre/_rtilde): (2.*TMath::Pi() - acos(_xerre/_rtilde));
  if(_phiR <= 0.)
    _phiR += 2 * TMath::Pi();
  if(_phiR > 2*TMath::Pi())
    _phiR -= 2 * TMath::Pi();
  _deltaPhiR = _phiR - _phiCMS; 
  
  // second approach ------------------------------------------------------------
  // .........setting _sqrtDscrm = 1 !!!
  // PLZ 041111 begin - modify PhiR calculation to account for sistematics effects 
  // close to Rtilde
  float _Xerre = _rtilde*cos(_alphaDT) + rho*sin(_alphaDT)*delta;
  float _Yerre = _rtilde*sin(_alphaDT) - rho*cos(_alphaDT)*delta;
  float _PhiR1 = (_Yerre>0.)? acos(_Xerre/_rtilde)
    :(2.*TMath::Pi() - acos(_Xerre/_rtilde));
  float _PhiR2 = (_Xerre>0.)? abs(asin(_Yerre/_rtilde))
    :(   TMath::Pi() - abs(asin(_Yerre/_rtilde)));
  //  cout << _Xerre << " " << _Yerre << " " << _PhiR2 << endl;
  if(_PhiR1 <= 0.)
    _PhiR1 += 2 * TMath::Pi();
  if(_PhiR1 > 2 * TMath::Pi())
    _PhiR1 -= 2 * TMath::Pi();
  if(_Yerre <0 )
    _PhiR2 = 2 * TMath::Pi()-_PhiR2;
  _PhiR = (_PhiR1+_PhiR2)/2;
  //    cout << " parziali " << _PhiR1 << " " << _PhiR2 << endl;
  // PLZ end

  // third approach -------------------------------------------------------------
  double zz = delta*_rho/_rtilde;
  bool accurate = pSet.getUntrackedParameter<bool>("third_method_accurate", false);
  _PhiRI = accurate? (_alphaDT  - zz - (1/6.)* zz*zz*zz):  (_alphaDT  - zz);

  if(_PhiRI <= 0.)
    _PhiRI += 2 * TMath::Pi();
  if(_PhiRI > 2*TMath::Pi())
    _PhiRI -= 2 * TMath::Pi();
  
  // cout << " Mu " << _Xerre << " " << _Yerre << " " << _PhiR << endl; 
  // cout << " mu " << _xerre << " " << _yerre << " " << _phiR <<  endl; 
  
  //------------------------------------------------------------------
  int i = -1;
  for(int l=0; l<StackedLayersInUseTotal; l++) {
    if( !_flagMatch[l] ) 
      continue;
    ++i;
  }
  _matching_stubs_No = i+1;
  
  for(size_t s=0; s<labels.size(); s++) {
    DTMatchPt* aPt = new DTMatchPt();
    
    if(labels[s].find(string("TrackletSL")) != string::npos)
      aPt = new DTMatchPt(labels[s], _station, pSet, _MatchingTkTracklets); 
    else if(labels[s].find(string("Mu_SL")) != string::npos) {
      aPt = new DTMatchPt(labels[s], _station, pSet, 
			  _xerre,_yerre,_MatchingTkTracklets, 
			  _stub_x, _stub_y,_flagMatch); 
    }
    else if(labels[s].find(string("mu")) != string::npos) // using proper _sqrtDscrm
      aPt = new DTMatchPt(labels[s], _station, pSet, 
			  // _rtilde*cos(_phiR), _rtilde*sin(_phiR), 
			  _xerre,_yerre,
			  _stub_x, _stub_y, _flagMatch); 
    else if( (labels[s].find(string("Mu")) != string::npos) &&
	     (labels[s].find(string("IMu")) == string::npos) ) // _sqrtDscrm set to 1
      aPt = new DTMatchPt(labels[s], _station, pSet, 
			  // _rtilde*cos(_PhiR), _rtilde*sin(_PhiR), 
			  _Xerre,_Yerre,
			  _stub_x, _stub_y, _flagMatch);
    else if(labels[s].find(string("IMu")) !=  string::npos) 
      // using virtual hit by linear extrapolation from DT;
      aPt = new DTMatchPt(labels[s], _station, pSet, 
			  _rtilde*cos(_PhiRI), _rtilde*sin(_PhiRI), 
			  _stub_x, _stub_y, _flagMatch);
    else if( (labels[s].find(string("Stubs")) != string::npos) && // all Stubs
	     (labels[s].find(string("LinStubs")) == string::npos) )
      aPt = new DTMatchPt(labels[s], _station, pSet, _stub_position, _flagMatch);
    else if(labels[s].find(string("LinStubs")) != string::npos) 
      aPt = new DTMatchPt(labels[s], _station, _stub_position, _flagMatch, pSet);
    else if(labels[s].find(string("LinFit")) != string::npos)  {
      // using linear fit of stub dephi vs invPt
      const int I = tracker_lay_Id_to_our( atoi( &((labels[s])[7]) ) );
      const int J = tracker_lay_Id_to_our( atoi( &((labels[s])[9]) ) );
      /*
	cout << "(" << I << ", " << J << ") --> flagMatches = "  << "(" 
	<< ((_flagMatch[I])? "true" : "false") << ", " 
	<< ((_flagMatch[J])? "true" : "false") <<  ")" << endl;
      */
      const float dephi = fabs(stubstub_dephi(I, J));
      const float slope = slope_linearfit(I, J);
      const float dephi_zero = y_intercept_linearfit(I, J);
      aPt = new DTMatchPt(labels[s], slope, dephi_zero, 
			      I, J, dephi, pSet, _flagMatch);
      /*
	cout << "(" << I << ", " << J << ") --> " << labels[s] 
	<< ": Pt = " << aPt->Pt() 
	<< endl << endl;
      */
    }
    // assign Pt object to appropriate DTMatchPtVariety data member:  
    assignPt(pSet, s, aPt);
  } 
}
  



