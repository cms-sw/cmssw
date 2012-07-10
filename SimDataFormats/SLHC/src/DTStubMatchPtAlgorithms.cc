#ifdef SLHC_DT_TRK_DFENABLE
#include <math.h>
#include <string>

#include "SimDataFormats/SLHC/interface/DTStubMatchPtAlgorithms.h"


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


static const double Ri_Coil = 315.0;
const double DTStubMatchPtAlgorithms::_RiCoil = Ri_Coil;
static const double Ro_Coil = 340.0;
const double DTStubMatchPtAlgorithms::_RoCoil = Ro_Coil;
const double DTStubMatchPtAlgorithms::_rtilde =
  (Ri_Coil*Ri_Coil + Ri_Coil*Ro_Coil + Ro_Coil*Ro_Coil)/(3.*Ro_Coil);



// constructor
DTStubMatchPtAlgorithms::DTStubMatchPtAlgorithms() {

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
}




// copy constructor
DTStubMatchPtAlgorithms::DTStubMatchPtAlgorithms(const DTStubMatchPtAlgorithms& pts):
  DTStubMatchPtVariety(pts)
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
  DTStubMatchPtAlgorithms&
  DTStubMatchPtAlgorithms::operator =(const DTStubMatchPtAlgorithms& pts) {
  if (this == &pts)      // Same object?
    return *this;        // Yes, so skip assignment, and just return *this.
  this->DTStubMatchPtVariety::operator=(pts);
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




void DTStubMatchPtAlgorithms::setPt(const edm::ParameterSet&  pSet)
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
  // Then get intercept of that straight line trajectory to the ideal cylindrical
  // boundary surface of the CMS magnetic field: this will be used as a point
  // belonging to the muon trajectory inside the magnetic field toghether with
  // already matched tracker stubs.
  // The idea id trying to avoid use of the outermost tracker layers (8th and 9nt),
  // stubs, however keeping comparable precision.

  // first approach ------------------------------------------------------------
  double rhosq = _position.x()*_position.x() + _position.y()*_position.y();
  double rho   = sqrt(rhosq);
  _rho = static_cast<float>(rho);
  double Dscrm  = 1. - rhosq*sin(delta)*sin(delta)/(_rtilde*_rtilde);
  if( Dscrm < 0.)
    return;
  _sqrtDscrm = static_cast<float>(sqrt(Dscrm));
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
  float _Xerre = _rtilde*cos(_alphaDT) + rho*sin(_alphaDT)*delta;
  float _Yerre = _rtilde*sin(_alphaDT) - rho*cos(_alphaDT)*delta;
  _PhiR = (_Yerre>0.)? acos(_Xerre/_rtilde): (2.*TMath::Pi() - acos(_Xerre/_rtilde));
  if(_PhiR <= 0.)
    _PhiR += 2 * TMath::Pi();
  if(_PhiR > 2*TMath::Pi())
    _PhiR -= 2 * TMath::Pi();

  // third approach -------------------------------------------------------------
  double zz = delta*_rho/_rtilde;
  bool accurate = pSet.getUntrackedParameter<bool>("third_method_accurate", false);
  _PhiRI = accurate? (_alphaDT  - zz - (1/6.)* zz*zz*zz):  (_alphaDT  - zz);

  if(_PhiRI <= 0.)
    _PhiRI += 2 * TMath::Pi();
  if(_PhiRI > 2*TMath::Pi())
    _PhiRI -= 2 * TMath::Pi();

  //------------------------------------------------------------------
  int i = -1;
  for(int l=0; l<StackedLayersInUseTotal; l++) {
    if( !_flagMatch[l] )
      continue;
    ++i;
  }
  _matching_stubs_No = i+1;

  for(size_t s=0; s<labels.size(); s++) {
    DTStubMatchPt* aPt = new DTStubMatchPt();
    if(labels[s].find(string("mu")) != string::npos) // using proper _sqrtDscrm
      aPt = new DTStubMatchPt(labels[s], _station, pSet,
			      _rtilde*cos(_phiR), _rtilde*sin(_phiR),
			      _stub_x, _stub_y, _flagMatch);
    else if( (labels[s].find(string("Mu")) != string::npos) &&
	     (labels[s].find(string("IMu")) == string::npos) ) // _sqrtDscrm set to 1
      aPt = new DTStubMatchPt(labels[s], _station, pSet,
			      _rtilde*cos(_PhiR), _rtilde*sin(_PhiR),
			      _stub_x, _stub_y, _flagMatch);
    else if(labels[s].find(string("IMu")) !=  string::npos)
      // using virtual hit by linear extrapolation from DT;
      aPt = new DTStubMatchPt(labels[s], _station, pSet,
			      _rtilde*cos(_PhiRI), _rtilde*sin(_PhiRI),
			      _stub_x, _stub_y, _flagMatch);
    else if( (labels[s].find(string("Stubs")) != string::npos) && // all Stubs
	     (labels[s].find(string("LinStubs")) == string::npos) )
      aPt = new DTStubMatchPt(labels[s], _station, pSet, _stub_position, _flagMatch);
    else if(labels[s].find(string("LinStubs")) != string::npos)
      aPt = new DTStubMatchPt(labels[s], _station, _stub_position, _flagMatch, pSet);
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
      aPt = new DTStubMatchPt(labels[s], slope, dephi_zero,
			      I, J, dephi, pSet, _flagMatch);
      /*
      cout << "(" << I << ", " << J << ") --> " << labels[s]
	   << ": Pt = " << aPt->Pt()
	   << endl << endl;
      */
    }
    // assign Pt object to appropriate DTStubMatch data member:
    assignPt(pSet, s, aPt);
  }
}




#endif
