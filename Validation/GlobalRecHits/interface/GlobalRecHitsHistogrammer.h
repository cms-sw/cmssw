#ifndef GlobalRecHitsHistogrammer_h
#define GlobalRecHitsHistogrammer_h

/** \class GlobalHitsProducer
 *  
 *  Class to fill PGlobalRecHit object to be inserted into data stream 
 *  containing information about various sub-systems in global coordinates 
 *  with full geometry
 *
 *  $Date: 2009/12/18 20:45:11 $
 *  $Revision: 1.6 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"


//#include "DataFormats/Common/interface/Provenance.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// event info
#include "SimDataFormats/ValidationFormats/interface/PValidationFormats.h"

// helper files
//#include <CLHEP/Vector/LorentzVector.h>
//#include <CLHEP/Units/SystemOfUnits.h>

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>
#include <map>
#include <math.h>

#include "TString.h"
#include "DQMServices/Core/interface/MonitorElement.h"


class GlobalRecHitsHistogrammer : public edm::EDAnalyzer
{

 public:

  //typedef std::vector<float> FloatVector;
  //typedef std::vector<double> DoubleVector;
  //typedef std::vector<int> IntVector;
  typedef std::map<uint32_t,float,std::less<uint32_t> > MapType;

  explicit GlobalRecHitsHistogrammer(const edm::ParameterSet&);
  virtual ~GlobalRecHitsHistogrammer();
  virtual void beginJob();
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  

 private:

  //  parameter information
  std::string fName;
  int verbosity;
  int frequency;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  DQMStore *dbe;
  std::string outputfile;
  bool doOutput;

  edm::InputTag GlobalRecHitSrc_;
  //edm::InputTag srcGlobalRecHits;

  // Electromagnetic info
  // ECal info
 
  MonitorElement *mehEcaln[3];
  MonitorElement *mehEcalRes[3];



  // HCal info

  MonitorElement *mehHcaln[4];
  MonitorElement *mehHcalRes[4];


  // Tracker info
  // SiStrip
  
  MonitorElement *mehSiStripn[19];
  MonitorElement *mehSiStripResX[19];
  MonitorElement *mehSiStripResY[19];


  // SiPxl

  MonitorElement *mehSiPixeln[7];
  MonitorElement *mehSiPixelResX[7];
  MonitorElement *mehSiPixelResY[7];

  // Muon info
  // DT

  MonitorElement *mehDtMuonn;
  MonitorElement *mehCSCn;
  MonitorElement *mehRPCn;
  MonitorElement *mehDtMuonRes;
  MonitorElement *mehCSCResRDPhi;
  MonitorElement *mehRPCResX;


  // private statistics information
  unsigned int count;

}; // end class declaration






#endif //PGlobalRecHitsProducer_h
