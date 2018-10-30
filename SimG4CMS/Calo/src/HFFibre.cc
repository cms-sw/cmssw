///////////////////////////////////////////////////////////////////////////////
// File: HFFibre.cc
// Description: Loads the table for attenuation length and calculates it
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFFibre.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include <iostream>
#include <sstream>

//#define DebugLog

HFFibre::HFFibre(const std::string & name, const DDCompactView & cpv, 
                 edm::ParameterSet const & p) {

  edm::ParameterSet m_HF = p.getParameter<edm::ParameterSet>("HFShower");
  cFibre           = c_light*(m_HF.getParameter<double>("CFibre"));
  
  edm::LogVerbatim("HFShower") << "HFFibre:: Speed of light in fibre " << cFibre
                           << " m/ns";

  std::string attribute = "Volume"; 
  std::string value     = "HF";
  DDSpecificsMatchesValueFilter filter1{DDValue(attribute,value,0)};
  DDFilteredView fv1(cpv,filter1);
  bool dodet = fv1.firstChild();

  if (dodet) {
    DDsvalues_type sv(fv1.mergedSpecifics());

    // Attenuation length
    nBinAtt      = -1;
    attL         = getDDDArray("attl",sv,nBinAtt);
    std::stringstream ss1;
    for (int it=0; it<nBinAtt; it++) {
      if(it/10*10 == it) { ss1 << "\n"; }
      ss1 << "  " << attL[it]*cm;
    }
    edm::LogVerbatim("HFShower") << "HFFibre: " << nBinAtt << " attL(1/cm): " << ss1.str();

    // Limits on Lambda
    int nb   = 2;
    std::vector<double>  nvec = getDDDArray("lambLim",sv,nb);
    lambLim[0] = static_cast<int>(nvec[0]);
    lambLim[1] = static_cast<int>(nvec[1]);
    edm::LogVerbatim("HFShower") << "HFFibre: Limits on lambda " << lambLim[0]
                             << " and " << lambLim[1];

    // Fibre Lengths
    nb       = 0;
    longFL   = getDDDArray("LongFL",sv,nb);
    std::stringstream ss2;
    for (int it=0; it<nb; it++) {
      if(it/10*10 == it) { ss2 << "\n"; }
      ss2 << "  " << longFL[it]/cm;
    }
    edm::LogVerbatim("HFShower") << "HFFibre: " << nb << " Long Fibre Length(cm):" << ss2.str();
    nb = 0;
    shortFL   = getDDDArray("ShortFL",sv,nb);
    std::stringstream ss3;
    for (int it=0; it<nb; it++) {
      if(it/10*10 == it) { ss3 << "\n"; }
      ss3 << "  " << shortFL[it]/cm;
    } 
    edm::LogVerbatim("HFShower") << "HFFibre: " << nb << " Short Fibre Length(cm):" << ss3.str();
  } else {
    edm::LogError("HFShower") << "HFFibre: cannot get filtered "
                              << " view for " << attribute << " matching "
                              << name;
    throw cms::Exception("Unknown", "HFFibre")
      << "cannot match " << attribute << " to " << name <<"\n";
  }
}

void HFFibre::initRun(const HcalDDDSimConstants* hcons) {

  // Now geometry parameters
  gpar      = hcons->getGparHF();
  radius    = hcons->getRTableHF();

  nBinR     = (int)(radius.size());
  std::stringstream sss;
  for (int i=0; i<nBinR; ++i) {
    if(i/10*10 == i) { sss << "\n"; }
    sss << "  " << radius[i]/cm;
  }
  edm::LogVerbatim("HFShower") << "HFFibre: " << radius.size() <<" rTable(cm):" << sss.str();
}

double HFFibre::attLength(double lambda) {

  int i = int(nBinAtt*(lambda - lambLim[0])/(lambLim[1]-lambLim[0]));

  int j =i;
  if (i >= nBinAtt) 
    j = nBinAtt-1;
  else if (i < 0)
    j = 0;
  double att = attL[j];
#ifdef DebugLog
  edm::LogVerbatim("HFShower") << "HFFibre::attLength for Lambda " << lambda
                           << " index " << i  << " " << j << " Att. Length " 
                           << att;
#endif
  return att;
}

double HFFibre::tShift(const G4ThreeVector& point, int depth, int fromEndAbs) {

  double zFibre = zShift(point, depth, fromEndAbs);
  double time   = zFibre/cFibre;
#ifdef DebugLog
  edm::LogVerbatim("HFShower") << "HFFibre::tShift for point " << point
                           << " ( depth = " << depth <<", traversed length = " 
                           << zFibre/cm  << " cm) = " << time/ns << " ns";
#endif
  return time;
}

double HFFibre::zShift(const G4ThreeVector& point, int depth, int fromEndAbs) { // point is z-local

  double zFibre = 0;
  double hR     = sqrt((point.x())*(point.x())+(point.y())*(point.y()));
  int    ieta   = 0;
  double length = 250*cm;
  if (fromEndAbs < 0) {
    zFibre = 0.5*gpar[1] - point.z(); // Never, as fromEndAbs=0 (?)
  } else {
    // Defines the Radius bin by radial subdivision
    for (int i = nBinR-1; i > 0; --i) if (hR < radius[i]) ieta = nBinR - i - 1;
    // define the length of the fibre
    if (depth == 2) {
      if ((int)(shortFL.size()) > ieta) length = shortFL[ieta];
    } else {
      if ((int)(longFL.size())  > ieta) length = longFL[ieta];
    }
    zFibre = length;
    if (fromEndAbs > 0) {
      zFibre   -= gpar[1]; // Never, as fromEndAbs=0 (M.K. ?)
    } else  {
      double zz = 0.5*gpar[1] + point.z();
      zFibre   -= zz;
    }
    if (depth == 2) zFibre += gpar[0]; // here zFibre is reduced for Short
  }

#ifdef DebugLog
  edm::LogVerbatim("HFShower") << "HFFibre::zShift for point " << point
                           << " (R = " << hR/cm << " cm, Index = " << ieta 
                           << ", depth = " << depth << ", Fibre Length = " 
                           << length/cm       << " cm = " << zFibre/cm  
                           << " cm)";
#endif
  return zFibre;
}

std::vector<double> HFFibre::getDDDArray(const std::string & str, 
                                         const DDsvalues_type & sv, 
                                         int & nmin) {

#ifdef DebugLog
  LogDebug("HFShower") << "HFFibre:getDDDArray called for " << str 
                       << " with nMin " << nmin;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("HFShower") << value;
#endif
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
      if (nval < 1 && nmin != 0) {
        edm::LogError("HFShower") << "HFFibre : # of " << str << " bins " 
                                  << nval << " < 1 ==> illegal (nmin=" 
                                  << nmin << ")";
        throw cms::Exception("Unknown", "HFFibre")
          << "nval < 1 for array " << str <<"\n";
      }
    }
    nmin = nval;
    return fvec;
  } else {
    if (nmin != 0) {
      edm::LogError("HFShower") << "HFFibre : cannot get array " << str;
      throw cms::Exception("Unknown", "HFFibre")
        << "cannot get array " << str <<"\n";
    } else {
      std::vector<double> fvec;
      return fvec;
    }
  }
}
