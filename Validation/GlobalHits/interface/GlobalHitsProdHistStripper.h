#ifndef GlobalHitsProdHistStripper_h
#define GlobalHitsProdHistStripper_h

/** \class GlobalHitsProdHistStripper
 *  
 *  Class to fill dqm monitor elements from existing EDM file
 *
 *  $Date: 2010/01/06 14:24:50 $
 *  $Revision: 1.7 $
 *  \author M. Strang SUNY-Buffalo
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// helper files
//#include <CLHEP/Vector/LorentzVector.h>
#include "DataFormats/Math/interface/LorentzVector.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

#include <iostream>
#include <stdlib.h>
#include <string>
#include <memory>
#include <vector>

#include "TString.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class GlobalHitsProdHistStripper : public edm::EDAnalyzer
{
  
 public:

  //typedef std::vector<float> FloatVector;

  explicit GlobalHitsProdHistStripper(const edm::ParameterSet&);
  virtual ~GlobalHitsProdHistStripper();
  virtual void beginJob( void );
  virtual void endJob();  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  
 private:

  //  parameter information
  std::string fName;
  int verbosity;
  int frequency;
  int vtxunit;
  bool getAllProvenances;
  bool printProvenanceInfo;


  DQMStore *dbe;
  std::string outputfile;
  bool doOutput;

  std::map<std::string,MonitorElement*> monitorElements;

  std::vector<MonitorElement*> me;

  // G4MC info
  MonitorElement *meMCRGP[2];
  MonitorElement *meMCG4Vtx[2];
  MonitorElement *meGeantVtxX[2];
  MonitorElement *meGeantVtxY[2];  
  MonitorElement *meGeantVtxZ[2];  
  MonitorElement *meMCG4Trk[2];
  MonitorElement *meGeantTrkPt;
  MonitorElement *meGeantTrkE; 

  // Electromagnetic info
  // ECal info
  MonitorElement *meCaloEcal[2];
  MonitorElement *meCaloEcalE[2];
  MonitorElement *meCaloEcalToF[2];
  MonitorElement *meCaloEcalPhi;
  MonitorElement *meCaloEcalEta;  

  // Preshower info
  MonitorElement *meCaloPreSh[2];
  MonitorElement *meCaloPreShE[2];
  MonitorElement *meCaloPreShToF[2];
  MonitorElement *meCaloPreShPhi;
  MonitorElement *meCaloPreShEta;

  // Hadronic info
  // HCal info
  MonitorElement *meCaloHcal[2];
  MonitorElement *meCaloHcalE[2];
  MonitorElement *meCaloHcalToF[2];
  MonitorElement *meCaloHcalPhi;
  MonitorElement *meCaloHcalEta;  

  // Tracker info
  // Pixel info
  //int nPxlBrlHits;
  //int nPxlFwdHits;
  //int nPxlHits;
  MonitorElement *meTrackerPx[2];
  MonitorElement *meTrackerPxPhi;
  MonitorElement *meTrackerPxEta;
  MonitorElement *meTrackerPxBToF;
  MonitorElement *meTrackerPxBR;
  MonitorElement *meTrackerPxFToF;
  MonitorElement *meTrackerPxFZ;

  // Strip info
  //int nSiHits;
  //int nSiBrlHits;
  //int nSiFwdHits;
  MonitorElement *meTrackerSi[2];
  MonitorElement *meTrackerSiPhi;
  MonitorElement *meTrackerSiEta;
  MonitorElement *meTrackerSiBToF;
  MonitorElement *meTrackerSiBR;
  MonitorElement *meTrackerSiFToF;
  MonitorElement *meTrackerSiFZ;

  // Muon info
  MonitorElement *meMuon[2];
  MonitorElement *meMuonPhi;
  MonitorElement *meMuonEta;
  //int nMuonHits;

  // DT info
  //int nMuonDtHits;
  MonitorElement *meMuonDtToF[2];
  MonitorElement *meMuonDtR;

  // CSC info
  //int nMuonCscHits;
  MonitorElement *meMuonCscToF[2];
  MonitorElement *meMuonCscZ;

  // RPC info
  //int nMuonRpcBrlHits;
  //int nMuonRpcFwdHits;
  MonitorElement *meMuonRpcFToF[2];
  MonitorElement *meMuonRpcFZ;
  MonitorElement *meMuonRpcBToF[2];
  MonitorElement *meMuonRpcBR;

  // private statistics information
  unsigned int count;

}; // end class declaration
  
#endif


