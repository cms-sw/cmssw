///////////////////////////////////////////////////////////////////////////////
// File: HFFibre.cc
// Description: Loads the table for attenuation length and calculates it
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFFibre.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/SystemOfUnits.h"
#include <iostream>

HFFibre::HFFibre(int iv, const DDCompactView & cpv) {

  verbosity = iv;
  std::string attribute = "Volume"; 
  std::string value     = "HF";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());

  // Attenuation length
  nBinAtt      = -1;
  attL         = getDDDArray("attl",sv,nBinAtt);
  if (verbosity > 0) {
    std::cout << "HFFibre: " << nBinAtt << " attL (1/cm)";
    for (int it=0; it<nBinAtt; it++) {
      std::cout << " " << attL[it]*cm;
      if (it%10 == 9) std::cout << std::endl << "                          ";
    }
    std::cout << std::endl;
  }

  // Limits on Lambda
  int             nbin = 2;
  std::vector<double>  nvec = getDDDArray("lambLim",sv,nbin);
  lambLim[0] = static_cast<int>(nvec[0]);
  lambLim[1] = static_cast<int>(nvec[1]);
  if (verbosity > 0) 
    std::cout << "HFFibre: Limits on lambda " << lambLim[0]
	      << " and " << lambLim[1] << std::endl;
}

HFFibre::~HFFibre() {}

double HFFibre::attLength(double lambda) {

  int i = int(nBinAtt*(lambda - lambLim[0])/(lambLim[1]-lambLim[0]));
  if (verbosity > 2) 
    std::cout << "HFFibre::attLength for Lambda " << lambda
	      << " index " << i;

  if (i >= nBinAtt) 
    i = nBinAtt-1;
  else if (i < 0)
    i = 0;
  double att = attL[i];
  if (verbosity > 2) 
    std::cout << " " << i << " Att. Length " << att << std::endl;

  return att;
}

std::vector<double> HFFibre::getDDDArray(const std::string & str, 
					 const DDsvalues_type & sv, 
					 int & nmin) {

  if (verbosity > 1) 
    std::cout << "HFFibre:getDDDArray called for " << str 
	      << " with nMin " << nmin << std::endl;
  DDValue value(str);
  if (DDfetch(&sv,value)) {
    if (verbosity > 2) std::cout << value << " " << std::endl;
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	if (verbosity > 0) 
	  std::cout << "HFFibre : # of " << str << " bins " << nval
		    << " < " << nmin << " ==> illegal" << std::endl;
	throw cms::Exception("Unknown", "HFFibre")
	  << "nval < nmin for array " << str <<"\n";
      }
    } else {
      if (nval < 2) {
	if (verbosity > 0) 
	  std::cout << "HFFibre : # of " << str << " bins " << nval
		    << " < 2 ==> illegal (nmin=" << nmin << ")" << std::endl;
	throw cms::Exception("Unknown", "HFFibre")
	  << "nval < 2 for array " << str <<"\n";
      }
    }
    nmin = nval;
    return fvec;
  } else {
    if (verbosity > 0) 
      std::cout << "HFFibre : cannot get array " << str << std::endl;
    throw cms::Exception("Unknown", "HFFibre")
      << "cannot get array " << str <<"\n";
  }
}
