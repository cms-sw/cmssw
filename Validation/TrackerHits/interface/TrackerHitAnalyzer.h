#ifndef TrackerHitAnalyzer_H
#define TrackerHitAnalyzer_H

/*
 * \file TrackerHitAnalyzer.h
 *
 * \author F. Cossutti
 *
*/
// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include <string>

class DQMStore;
class MonitorElement;

class TrackerHitAnalyzer: public edm::EDAnalyzer {
  
public:

/// Constructor
TrackerHitAnalyzer(const edm::ParameterSet& ps);

/// Destructor
~TrackerHitAnalyzer();

protected:

/// Begin Run
void beginRun( edm::Run const&, edm::EventSetup const&);

/// Analyze
void analyze(const edm::Event& e, const edm::EventSetup& c);



// EndJob
void endJob(void);

//void BookTestHistos(Char_t sname, int nbin, float *xmin, float *xmax);

private:

 bool verbose_;

 edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_pxlBrlLow_Token_, edmPSimHitContainer_pxlBrlHigh_Token_;
 edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_pxlFwdLow_Token_, edmPSimHitContainer_pxlFwdHigh_Token_;
 edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_siTIBLow_Token_, edmPSimHitContainer_siTIBHigh_Token_;
 edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_siTOBLow_Token_, edmPSimHitContainer_siTOBHigh_Token_;
 edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_siTIDLow_Token_, edmPSimHitContainer_siTIDHigh_Token_;
 edm::EDGetTokenT<edm::PSimHitContainer> edmPSimHitContainer_siTECLow_Token_, edmPSimHitContainer_siTECHigh_Token_;
 edm::EDGetTokenT<edm::SimTrackContainer> edmSimTrackContainerToken_;

 DQMStore* fDBE;

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

 std::string fOutputFile;
};

#endif
