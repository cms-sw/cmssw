// -*- C++ -*-
//
// Package:    PhotonJetAnalyzer
// Class:      PhotonJetAnalyzer
// 
/**\class PhotonJetAnalyzer PhotonJetAnalyzer.cc Analysis/PhotonJetAnalyzer/src/PhotonJetAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Mike Anderson
//         Created:  Tue Apr 22 10:19:02 CDT 2008
// $Id: PhotonJetAnalyzer.h,v 1.1 2009/11/10 18:52:45 rwolf Exp $
//
//


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"

// ROOT headers
#include "TFile.h"
#include "TH1.h"
#include "TH1F.h"


class PhotonJetAnalyzer : public edm::EDAnalyzer {
  public:
    explicit PhotonJetAnalyzer(const edm::ParameterSet&);
    ~PhotonJetAnalyzer();


  private:
    virtual void beginJob() ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;

  // ----------member data ---------------------------

  // Basic set of variables to store for physics objects                                                                   
  struct basicStruct {
    double energy;
    double et;
    double pt;
    double eta;
    double phi;
  } ;

  // Functions
  double calcDeltaR(double eta1, double phi1, double eta2, double phi2);
  double calcDeltaPhi(double phi1, double phi2);

  // ***** Simple Histograms *****
  // Generated Photon
  TH1F*       hGenPhtnHardEt;
  TH1F*       hGenPhtnHardEta;
  TH1F*       hGenPhtnHardPhi;
  TH1F*       hGenPhtnHardMom;
  TH1F*       hGenPhtnHardDrJet;

  // Generated Jet
  TH1F*       hHIGenJetPt;
  TH1F*       hHIGenJetEta;
  TH1F*       hHIGenJetPhi;
  TH1F*       hHIGenJetCnt;
};
