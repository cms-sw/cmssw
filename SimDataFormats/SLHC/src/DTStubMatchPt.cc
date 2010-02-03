#include <math.h>

//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatch.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatchPt.h"
#include "SimDataFormats/SLHC/interface/DTStubMatch.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchPt.h"
#include "SLHCUpgradeSimulations/L1Trigger/interface/DTParameters.h"    

/**************************************************************************
  Only stubs and (primary) vertex
***************************************************************************/

void DTStubMatchPt::radius_of_curvature(const edm::ParameterSet& pSet,
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
  /*
  if( _bending < min_bending || _bending > max_bending ) {
    _Rb = _invRb = _bending = NAN;
    return;
  }
  */
  // _Rb = static_cast<float>(0.5 * sqrt(2*L1DotL2 + L1sq + L2sq)/sin(delta));
  _invRb = 2. * static_cast<float>(delta/(L1+L2));
  if( _invRb < min_invRb || _invRb > max_invRb ) {
    _Rb = _invRb = NAN; 
    return; 
  }
  _Rb = static_cast<float>(0.5*(L1+L2)/delta);
} 
 


void DTStubMatchPt::setPt(const edm::ParameterSet& pSet,
			  float const X[], float const Y[], float const corr) {
  radius_of_curvature(pSet, X, Y);
  if( isnan( _Rb ) )
    return;
  _Pt = 0.003 * _Rb * 3.8;
  _Pt += corr * _Pt;
  _invPt = 1./_Pt;
  return;
} 



DTStubMatchPt::DTStubMatchPt(string const s, int station,
			     const edm::ParameterSet& pSet, 
			     float const stub_x[], float const stub_y[], 
			     bool const flagMatch[]):
  _label(s)  
{
  float X[3], Y[3];  

  _Rb = _invRb = _Pt = _invPt = NAN;

  if( _label == string("Stubs-5-3-0") && flagMatch[5] && flagMatch[3] && flagMatch[0] ) {
    X[0] = stub_x[5];
    Y[0] = stub_y[5];
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Stubs-5-2-0") && flagMatch[5] && flagMatch[2] && flagMatch[0] ) {
    X[0] = stub_x[5];
    Y[0] = stub_y[5];
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Stubs-5-1-0") && flagMatch[5] && flagMatch[1] && flagMatch[0] ) {
    X[0] = stub_x[5];
    Y[0] = stub_y[5];
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Stubs-3-2-0") && flagMatch[3] && flagMatch[2] && flagMatch[0] ) {
    X[0] = stub_x[3];
    Y[0] = stub_y[3];
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Stubs-3-1-0") && flagMatch[3] && flagMatch[1] && flagMatch[0] ) {
    X[0] = stub_x[3];
    Y[0] = stub_y[3];
    X[1] = stub_x[1];
    Y[1] = stub_y[1]; 
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  //******************
  // Next use vertex:
  //******************

  X[2] = 0.;
  Y[2] = 0.;

  if( _label == string("Stubs-5-3-V") && flagMatch[5] && flagMatch[3] ) { 
    X[0] = stub_x[5];
    Y[0] = stub_y[5];
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Stubs-5-0-V") && flagMatch[5] && flagMatch[0] ) { 
    X[0] = stub_x[5];
    Y[0] = stub_y[5];
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Stubs-3-0-V") && flagMatch[3] && flagMatch[0] ) { 
    X[0] = stub_x[3];
    Y[0] = stub_y[3];
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
}



/**************************************************************************
   Using DT muons
***************************************************************************/

DTStubMatchPt::DTStubMatchPt(string const s, int station,
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

  _Rb = _invRb = _Pt = _invPt = NAN;

  if(isnan(DTmu_x) || isnan(DTmu_y)) 
    return;

  X[0] = DTmu_x;
  Y[0] = DTmu_y;

  if( _label == string("Mu-5-0") && flagMatch[5] && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu-3-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu-2-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu-1-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu-5-V") && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu-3-V") && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu-2-V") && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu-1-V") && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("Mu-0-V") && flagMatch[0] ) {
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-5-0") && flagMatch[5] && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-3-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-2-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-1-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-5-V") && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-3-V") && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-2-V") && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-1-V") && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("IMu-0-V") && flagMatch[0] ) {
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-5-0") && flagMatch[5] && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-3-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-2-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-1-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-5-V") && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-3-V") && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-2-V") && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-1-V") && flagMatch[0] ) {
    X[1] = stub_x[1];
    Y[1] = stub_y[1];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }
  if( _label == string("mu-0-V") && flagMatch[0] ) {
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y, corr);
    return;
  }

  /*
  if( _label == string("Zo-5-0") && flagMatch[5] && flagMatch[0] ) {
    X[1] = stub_x[5];
    Y[1] = stub_y[5];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Zo-3-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[3];
    Y[1] = stub_y[3];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Zo-2-0") && flagMatch[3] && flagMatch[0] ) {
    X[1] = stub_x[2];
    Y[1] = stub_y[2];
    X[2] = stub_x[0];
    Y[2] = stub_y[0];
    setPt(pSet, X, Y);
    return;
  }
  if( _label == string("Zo-0-V") && flagMatch[0] ) {
    X[1] = stub_x[0];
    Y[1] = stub_y[0];
    X[2] = 0.;
    Y[2] = 0.;
    setPt(pSet, X, Y);
    return;
  }
  */
}




/**************************************************************************
   Only DT muons
***************************************************************************/

DTStubMatchPt::DTStubMatchPt(int station,
			     const edm::ParameterSet& pSet, 
			     float const bendingDT, float const Rb) 
{
  float corr = 0.0; //0.06;
  _invRb = _Rb = _Pt = _invPt = NAN;

  float min_invRb = pSet.getUntrackedParameter<double>("min_invRb");
  float max_invRb = pSet.getUntrackedParameter<double>("max_invRb");

  if( isnan(Rb) )
    return;
  _Rb = Rb;
  _invRb = 1./Rb; 
  if( _invRb < min_invRb || _invRb > max_invRb ) {
    _Rb = _invRb = NAN;
    return;
  }

  _Pt = 0.003 * Rb * 3.8;
  _Pt += corr * _Pt;
  _invPt = 1./_Pt;

  return;
} 

