#ifndef PhotonPostprocessing_H
#define PhotonPostprocessing_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
//


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

//

#include <vector>

/** \class PhotonPostprocessing
 **  
 **
 **  $Id: PhotonPostprocessing
 **  $Date: 2013/06/07 11:21:54 $ 
 **  authors: 
 **   Nancy Marinelli, U. of Notre Dame, US  
 **     
 ***/


// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;


class PhotonPostprocessing : public edm::EDAnalyzer
{

 public:
   
  //
  explicit PhotonPostprocessing( const edm::ParameterSet& pset ) ;
  virtual ~PhotonPostprocessing();
                                   
      
  virtual void analyze(const edm::Event&, const edm::EventSetup&  ) ;
  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void endLuminosityBlock( const edm::LuminosityBlock& , const edm::EventSetup& ) ;
  virtual void endRun(const edm::Run& , const edm::EventSetup& ) ;
 

 private:
  //



  void dividePlots(MonitorElement* dividend, MonitorElement* numerator, MonitorElement* denominator,std::string type);
  void dividePlots(MonitorElement* dividend, MonitorElement* numerator, double denominator); 
  virtual void runPostprocessing();      

  DQMStore *dbe_;
  int verbosity_;

  edm::ParameterSet parameters_;


  bool standAlone_;
  bool batch_;
  bool isRunCentrally_;
  bool fastSim_;
  std::string analyzerName_;
  std::string outputFileName_;
  std::string inputFileName_;

  std::stringstream currentFolder_;


  double etMin;
  double etMax;
  int    etBin;
  double etaMin;
  double etaMax;
  int    etaBin;
  int    etaBin2;
  double phiMin;
  double phiMax;
  int    phiBin;
  double rMin;
  double rMax;
  int    rBin;
  double zMin;
  double zMax;
  int    zBin;



  MonitorElement*  phoRecoEffEta_;
  MonitorElement*  phoRecoEffPhi_;
  MonitorElement*  phoRecoEffEt_;

  MonitorElement*  phoDeadChEta_;
  MonitorElement*  phoDeadChPhi_;
  MonitorElement*  phoDeadChEt_;


  MonitorElement*  convEffEtaTwoTracks_;
  MonitorElement*  convEffPhiTwoTracks_;
  MonitorElement*  convEffRTwoTracks_;
  MonitorElement*  convEffZTwoTracks_;
  MonitorElement*  convEffEtTwoTracks_;

  MonitorElement*  convEffEtaTwoTracksAndVtxProbGT0_;
  MonitorElement*  convEffEtaTwoTracksAndVtxProbGT0005_;
  MonitorElement*  convEffRTwoTracksAndVtxProbGT0_;
  MonitorElement*  convEffRTwoTracksAndVtxProbGT0005_;

  MonitorElement*  convEffEtaOneTrack_;
  MonitorElement*  convEffROneTrack_;
  MonitorElement*  convEffEtOneTrack_;

  MonitorElement*  convFakeRateEtaTwoTracks_;
  MonitorElement*  convFakeRatePhiTwoTracks_;
  MonitorElement*  convFakeRateRTwoTracks_;
  MonitorElement*  convFakeRateZTwoTracks_;
  MonitorElement*  convFakeRateEtTwoTracks_;

  MonitorElement*  bkgRecoEffEta_;
  MonitorElement*  bkgRecoEffPhi_;
  MonitorElement*  bkgRecoEffEt_;

  MonitorElement*  bkgDeadChEta_;
  MonitorElement*  bkgDeadChPhi_;
  MonitorElement*  bkgDeadChEt_;

  MonitorElement*  convVsEt_[2];

   
};





#endif
