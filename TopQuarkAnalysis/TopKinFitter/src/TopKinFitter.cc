#include "TopQuarkAnalysis/TopKinFitter/interface/TopKinFitter.h"

/// default configuration is: max iterations = 200, max deltaS = 5e-5, maxF = 1e-4
TopKinFitter::TopKinFitter(const int maxNrIter, const double maxDeltaS, const double maxF,
			   const double mW, const double mTop): 
  maxNrIter_(maxNrIter), maxDeltaS_(maxDeltaS), maxF_(maxF), mW_(mW), mTop_(mTop)
{
  fitter_ = new TKinFitter("TopKinFitter", "TopKinFitter");
  fitter_->setMaxNbIter(maxNrIter_);
  fitter_->setMaxDeltaS(maxDeltaS_);
  fitter_->setMaxF(maxF_);
  fitter_->setVerbosity(0);
}

/// default destructor
TopKinFitter::~TopKinFitter() 
{
  delete fitter_;
}

/// convert Param to human readable form
std::string 
TopKinFitter::param(const Param& param) const
{
  std::string parName;
  switch(param){
  case kEMom       : parName="EMom";       break;
  case kEtEtaPhi   : parName="EtEtaPhi";   break;
  case kEtThetaPhi : parName="EtThetaPhi"; break;    
  }
  return parName;
}
