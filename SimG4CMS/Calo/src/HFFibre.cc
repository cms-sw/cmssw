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

HFFibre::HFFibre(const DDCompactView & cpv) {

  std::string attribute = "Volume"; 
  std::string value     = "HF";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();

  if (dodet) {
    DDsvalues_type sv(fv.mergedSpecifics());

    // Attenuation length
    nBinAtt      = -1;
    attL         = getDDDArray("attl",sv,nBinAtt);
    edm::LogInfo("HFShower") << "HFFibre: " << nBinAtt << " attL (1/cm)";
    for (int it=0; it<nBinAtt; it++) 
      edm::LogInfo("HFShower") << "HFFibre: attL[" << it << "] = " 
			       << attL[it]*cm;

    // Limits on Lambda
    int             nbin = 2;
    std::vector<double>  nvec = getDDDArray("lambLim",sv,nbin);
    lambLim[0] = static_cast<int>(nvec[0]);
    lambLim[1] = static_cast<int>(nvec[1]);
    edm::LogInfo("HFShower") << "HFFibre: Limits on lambda " << lambLim[0]
			     << " and " << lambLim[1];
  }
}

HFFibre::~HFFibre() {}

double HFFibre::attLength(double lambda) {

  int i = int(nBinAtt*(lambda - lambLim[0])/(lambLim[1]-lambLim[0]));

  int j =i;
  if (i >= nBinAtt) 
    j = nBinAtt-1;
  else if (i < 0)
    j = 0;
  double att = attL[j];
  LogDebug("HFShower") << "HFFibre::attLength for Lambda " << lambda
		       << " index " << i  << " " << j << " Att. Length " 
		       << att;

  return att;
}

std::vector<double> HFFibre::getDDDArray(const std::string & str, 
					 const DDsvalues_type & sv, 
					 int & nmin) {

  LogDebug("HFShower") << "HFFibre:getDDDArray called for " << str 
		       << " with nMin " << nmin;
  DDValue value(str);
  if (DDfetch(&sv,value)) {
    LogDebug("HFShower") << value;
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
	edm::LogError("HFShower") << "HFFibre : # of " << str << " bins " 
				  << nval << " < " << nmin << " ==> illegal";
	throw cms::Exception("Unknown", "HFFibre")
	  << "nval < nmin for array " << str <<"\n";
      }
    } else {
      if (nval < 2) {
	edm::LogError("HFShower") << "HFFibre : # of " << str << " bins " 
				  << nval << " < 2 ==> illegal (nmin=" 
				  << nmin << ")";
	throw cms::Exception("Unknown", "HFFibre")
	  << "nval < 2 for array " << str <<"\n";
      }
    }
    nmin = nval;
    return fvec;
  } else {
    edm::LogError("HFShower") << "HFFibre : cannot get array " << str;
    throw cms::Exception("Unknown", "HFFibre")
      << "cannot get array " << str <<"\n";
  }
}
