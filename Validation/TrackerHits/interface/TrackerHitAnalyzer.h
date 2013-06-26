#ifndef TrackerHitAnalyzer_H
#define TrackerHitAnalyzer_H

/*
 * \file TrackerHitAnalyzer.h
 *
 * $Date: 2012/09/04 21:50:26 $
 * $Revision: 1.7 $
 * \author F. Cossutti
 *
*/
// framework & common header files
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include <fstream>
#include <map>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>


class TrackerHitAnalyzer: public edm::EDAnalyzer {
  
public:

/// Constructor
TrackerHitAnalyzer(const edm::ParameterSet& ps);

/// Destructor
~TrackerHitAnalyzer();

protected:

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);



// EndJob
void endJob(void);

//void BookTestHistos(Char_t sname, int nbin, float *xmin, float *xmax);

private:

  edm::InputTag SiTIBLowSrc_;
  edm::InputTag SiTIBHighSrc_;
  edm::InputTag SiTOBLowSrc_;
  edm::InputTag SiTOBHighSrc_;
  edm::InputTag SiTIDLowSrc_;
  edm::InputTag SiTIDHighSrc_;
  edm::InputTag SiTECLowSrc_;
  edm::InputTag SiTECHighSrc_;
  edm::InputTag PxlBrlLowSrc_;
  edm::InputTag PxlBrlHighSrc_;
  edm::InputTag PxlFwdLowSrc_;
  edm::InputTag PxlFwdHighSrc_;
  edm::InputTag G4TrkSrc_;

//  edm::ParameterSet config_;
 
 
 bool verbose_;
 
 DQMStore* fDBE;
 
 std::string fOutputFile;

 MonitorElement* htofeta;
 MonitorElement* htofphi;
 MonitorElement* htofr;
 MonitorElement* htofz;
 MonitorElement* htofeta_profile;
 MonitorElement* htofphi_profile;
 MonitorElement* htofr_profile;
 MonitorElement* htofz_profile;
 MonitorElement* h1e[12];
 MonitorElement* h2e[12];
 MonitorElement* h3e[12];
 MonitorElement* h4e[12];
 MonitorElement* h5e[12];
 MonitorElement* h6e[12];
 
 MonitorElement* h1ex[12];
 MonitorElement* h2ex[12];
 MonitorElement* h3ex[12];
 MonitorElement* h4ex[12];
 MonitorElement* h5ex[12];
 MonitorElement* h6ex[12];
 
 MonitorElement* h1ey[12];
 MonitorElement* h2ey[12];
 MonitorElement* h3ey[12];
 MonitorElement* h4ey[12];
 MonitorElement* h5ey[12];
 MonitorElement* h6ey[12];
 
 MonitorElement* h1ez[12];
 MonitorElement* h2ez[12];
 MonitorElement* h3ez[12];
 MonitorElement* h4ez[12];
 MonitorElement* h5ez[12];
 MonitorElement* h6ez[12];

 MonitorElement* h1lx[12];
 MonitorElement* h2lx[12];
 MonitorElement* h3lx[12];
 MonitorElement* h4lx[12];
 MonitorElement* h5lx[12];
 MonitorElement* h6lx[12];
 
 MonitorElement* h1ly[12];
 MonitorElement* h2ly[12];
 MonitorElement* h3ly[12];
 MonitorElement* h4ly[12];
 MonitorElement* h5ly[12];
 MonitorElement* h6ly[12];
};

#endif
