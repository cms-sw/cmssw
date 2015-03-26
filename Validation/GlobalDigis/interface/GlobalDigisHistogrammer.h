#ifndef GlobalDigisHistogrammer_h
#define GlobalDigisHistogrammer_h

/** \class GlobalHitsProducer
 *  
 *  Class to fill PGlobalDigis object to be inserted into data stream 
 *  containing information about various sub-systems in global coordinates 
 *  with full geometry
 *
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

//DQM services
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
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


class GlobalDigisHistogrammer : public DQMEDAnalyzer {

 public:

  //typedef std::vector<float> FloatVector;
  //typedef std::vector<double> DoubleVector;
  //typedef std::vector<int> IntVector;
  typedef std::map<uint32_t,float,std::less<uint32_t> > MapType;

  explicit GlobalDigisHistogrammer(const edm::ParameterSet&);
  virtual ~GlobalDigisHistogrammer();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  void bookHistograms(DQMStore::IBooker &,
      edm::Run const &, edm::EventSetup const &) override;
  

 private:

  //  parameter information
  std::string fName;
  int verbosity;
  int frequency;
  std::string label;
  bool getAllProvenances;
  bool printProvenanceInfo;

  std::string outputfile;
  bool doOutput;

  edm::InputTag GlobalDigisSrc_;
  //edm::InputTag srcGlobalDigis;
  edm::EDGetTokenT<PGlobalDigi> GlobalDigisSrc_Token_;

  // Electromagnetic info
  // ECal info
  MonitorElement *mehEcaln[3];
  MonitorElement *mehEcalAEE[2];
  MonitorElement *mehEcalSHE[2];
  MonitorElement *mehEcalMaxPos[2];
  MonitorElement *mehEcalMultvAEE[2];
  MonitorElement *mehEcalSHEvAEESHE[2];
  MonitorElement *mehEScalADC[3];

  // HCal info
  MonitorElement *mehHcaln[4];
  MonitorElement *mehHcalAEE[4];
  MonitorElement *mehHcalSHE[4];
  MonitorElement *mehHcalAEESHE[4];
  MonitorElement *mehHcalSHEvAEE[4];


  // Tracker info
  // SiStrip
  
  MonitorElement *mehSiStripn[19];
  MonitorElement *mehSiStripADC[19];
  MonitorElement *mehSiStripStrip[19];


  // SiPxl

   MonitorElement *mehSiPixeln[7];
  MonitorElement *mehSiPixelADC[7];
  MonitorElement *mehSiPixelRow[7];
  MonitorElement *mehSiPixelCol[7];

  // Muon info
  // DT
  MonitorElement *mehDtMuonn[4];
  MonitorElement *mehDtMuonLayer[4];
  MonitorElement *mehDtMuonTime[4];
  MonitorElement *mehDtMuonTimevLayer[4];

  // CSC Strip

  MonitorElement *mehCSCStripn;
  MonitorElement *mehCSCStripADC;
  MonitorElement *mehCSCWiren;
  MonitorElement *mehCSCWireTime;

  float theCSCStripPedestalSum;
  int theCSCStripPedestalCount;
  // private statistics information
  unsigned int count;

}; // end class declaration






#endif //PGlobalDigisHistogrammer_h
