#include "SimRomanPot/CTPPSOpticsParameterisation/interface/TMultiDimFet.h"

void
TMultiDimFet::FindParameterization( double precision )
{
  // Find the parameterization
  MakeNormalized();
  MakeCandidates();
  MakeParameterization();
  MakeCoefficients();
  //MakeCoefficientErrors();
  //MakeCorrelation();
  ReducePolynomial( precision );
  ReleaseMemory();
}

void
TMultiDimFet::ReleaseMemory()
{
  fFunctions.ResizeTo(1,1);
  fOrthFunctions.ResizeTo(1,1);
  fOrthFunctionNorms.ResizeTo(1);
  fOrthCoefficients.ResizeTo(1);
  fOrthCurvatureMatrix.ResizeTo(1,1);
  for ( int i=0; i<fNVariables; ++i ) {
    fFunctionCodes[i] = 0;
  }
  //fFunctionCodes.resize(1);
}

void
TMultiDimFet::ReducePolynomial( double error )
{
  if ( error==0.0 ) return;
  else ZeroDoubiousCoefficients( error );
}

void
TMultiDimFet::ZeroDoubiousCoefficients( double error )
{
  typedef std::multimap<double, int> cmt;
  cmt m;

  for ( int i=0; i<fNCoefficients; i++ ) {
    m.insert(std::pair<double, int>(TMath::Abs(fCoefficients(i)), i));
  }
  double del_error_abs = 0.;
  int deleted_terms_count = 0;

  for ( cmt::iterator it=m.begin(); it!=m.end() && del_error_abs<error; ++it ) {
    if ( TMath::Abs(it->first)+del_error_abs<error ) {
      fCoefficients(it->second) = 0.;
      del_error_abs = TMath::Abs(it->first)+del_error_abs;
      deleted_terms_count++;
    }
    else break;
  }
  int fNCoefficients_new = fNCoefficients-deleted_terms_count;
  TVectorD fCoefficients_new( fNCoefficients_new );
  std::vector<int> fPowerIndex_new;

  int ind=0;
  for ( int i=0; i<fNCoefficients; i++ ) {
    if ( fCoefficients(i)==0. ) continue;
    fCoefficients_new(ind)=fCoefficients(i);
    fPowerIndex_new.push_back(fPowerIndex[i]);
    ind++;
  }

  fNCoefficients = fNCoefficients_new;
  fCoefficients.ResizeTo(fNCoefficients);
  fCoefficients = fCoefficients_new;
  //fPowerIndex = fPowerIndex_new;
  std::copy( fPowerIndex_new.begin(), fPowerIndex_new.end(), fPowerIndex );

  //std::cout<<deleted_terms_count<<" terms removed"<<std::endl;
}

