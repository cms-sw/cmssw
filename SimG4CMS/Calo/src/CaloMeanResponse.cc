#include "SimG4CMS/Calo/interface/CaloMeanResponse.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <fstream>
#include <iomanip>

#define DebugLog

CaloMeanResponse::CaloMeanResponse(edm::ParameterSet const & p) {

  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("CaloResponse");
  edm::FileInPath fp    = m_p.getParameter<edm::FileInPath>("ResponseFile");
  std::string fName     = fp.fullPath();
  useTable              = m_p.getParameter<bool>("UseResponseTable");
  scale                 = m_p.getParameter<double>("ResponseScale");
  edm::LogInfo("CaloSim") << "CaloMeanResponse initialized with scale " 
			  << scale << " and use Table " << useTable 
			  << " from file " << fName;

  readResponse (fName);
}

CaloMeanResponse::~CaloMeanResponse() {}

double CaloMeanResponse::getWeight(int genPID, double genP) {
  double weight = 1;
  bool   found=false;
  for (unsigned int i=0; i<pionTypes.size(); i++) {
    if (genPID == pionTypes[i]) {
      found = true;
      break;
    }
  }
  if (found) {
    weight = scale;
    if (useTable) {
      if (piLast >= 0) weight *= pionTable[piLast];
      for (unsigned int i=0; i<pionTable.size(); i++) {
	if (genP < pionMomentum[i]) {
	  if (i == 0) weight = scale*pionTable[i];
	  else        weight = scale*((pionTable[i-1]*(pionMomentum[i]-genP)+
				       pionTable[i]*(genP-pionMomentum[i-1]))/
				      (pionMomentum[i]-pionMomentum[i-1]));
	  break;
	}
      }
    }
#ifdef DebugLog
    LogDebug("CaloSim") << "CaloMeanResponse::getWeight PID " << genPID
			<< " uses pion list and gets weight " << weight
			<< " for momentum " << genP/GeV << " GeV/c";
#endif
  } else {
    for (unsigned int i=0; i<protonTypes.size(); i++) {
      if (genPID == protonTypes[i]) {
	found = true;
	break;
      }
    }
    if (found) {
      weight = scale;
      if (useTable) {
	if (pLast >= 0) weight *= protonTable[pLast];
	for (unsigned int i=0; i<protonTable.size(); i++) {
	  if (genP < protonMomentum[i]) {
	    if (i == 0) weight = scale*protonTable[i];
	    else        weight = scale*((protonTable[i-1]*(protonMomentum[i]-genP)+
					 protonTable[i]*(genP-protonMomentum[i-1]))/
					(protonMomentum[i]-protonMomentum[i-1]));
	    break;
	  }
	}
      }
#ifdef DebugLog
      LogDebug("CaloSim") << "CaloMeanResponse::getWeight PID " << genPID
			  << " uses proton list and gets weight " << weight
			  << " for momentum " << genP/GeV << " GeV/c";
#endif
    } else {
#ifdef DebugLog
      LogDebug("CaloSim") << "CaloMeanResponse::getWeight PID " << genPID
			  << " is not in either lists and weight " << weight;
#endif
    }
  }
  return weight;
}

void CaloMeanResponse::readResponse (std::string fName) {

  std::ifstream infile;
  infile.open(fName.c_str(), std::ios::in);

  if (infile) {
    int    nene, npart, pid;
    double ene, responseData, responseMC, ratio;

    // First read the pion data
    infile >> nene >> npart;
    for (int i=0; i<npart; i++) {
      infile >> pid;
      pionTypes.push_back(pid);
    }
    for (int i=0; i<nene; i++) {
      infile >> ene >> responseData >> responseMC;
      if (responseMC > 0) ratio = responseData/responseMC;
      else                ratio = 1;
      pionMomentum.push_back(ene*GeV);
      pionTable.push_back(ratio);
    }

    // Then read the proton data
    infile >> nene >> npart;
    for (int i=0; i<npart; i++) {
      infile >> pid;
      protonTypes.push_back(pid);
    }
    for (int i=0; i<nene; i++) {
      infile >> ene >> responseData >> responseMC;
      if (responseMC > 0) ratio = responseData/responseMC;
      else                ratio = 1;
      protonMomentum.push_back(ene*GeV);
      protonTable.push_back(ratio);
    }
    infile.close();
  }

  piLast = (int)(pionTable.size()) - 1;
  pLast  = (int)(protonTable.size()) - 1;
#ifdef DebugLog
  LogDebug("CaloSim") << "CaloMeanResponse::readResponse finds "
		      << pionTypes.size() << " particles to use pion response"
		      << " map with a table of " << pionTable.size() 
		      << " data points " << piLast;
  for (unsigned int i=0; i<pionTypes.size(); i++) 
    LogDebug("CaloSim") << "Particle ID[" << i << "] = " << pionTypes[i];
  for (unsigned int i=0; i<pionTable.size(); i++) 
    LogDebug("CaloSim") << "Momentum[" << i << "] (" << pionMomentum[i]/GeV
			<< " GeV/c) --> " << pionTable[i];
  LogDebug("CaloSim") << "CaloMeanResponse::readResponse finds "
		      << protonTypes.size() << " particles to use proton "
		      << "response map with a table of " << protonTable.size()
		      << " data points " << pLast;
  for (unsigned int i=0; i<protonTypes.size(); i++) 
    LogDebug("CaloSim") << "Particle ID[" << i << "] = " << protonTypes[i];
  for (unsigned int i=0; i<protonTable.size(); i++) 
    LogDebug("CaloSim") << "Momentum[" << i << "] (" << protonMomentum[i]/GeV
			<< " GeV/c) --> " << protonTable[i];
#endif
}
