// -*- C++ -*-
//
// Package:    AnalyzeTuples
// Class:      AnalyzeTuples
// 

/**\class AnalyzeTuples AnalyzeTuples.cc Analysis/AnalyzeTuples/src/AnalyzeTuples.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
// 
// Original Author: Taylan Yetkin
// Created: Tue Feb 10 08:43:07 CST 2009
// $Id: AnalyzeTuples.h,v 1.4 2010/02/11 00:14:58 wmtan Exp $
// 
// 


#include <memory>
#include <vector> 
#include <string> 

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TH1I.h"

class AnalyzeTuples:public edm::EDAnalyzer {
  public:
    explicit AnalyzeTuples(const edm::ParameterSet &);
    ~AnalyzeTuples();


  private:
    virtual void beginJob();
    virtual void analyze(const edm::Event &, const edm::EventSetup &);
    virtual void endJob();
    void loadEventInfo(TBranch *);
    void getRecord(int type, int record);
    TFile* hf;
    TBranch *emBranch, *hadBranch;
    
    int nMomBin, totEvents, evtPerBin;
    float libVers, listVersion; 
    std::vector<double> pmom;
    std::vector<HFShowerPhoton> photon;
    
    edm::Service < TFileService > fs;
    TH1I* hNPELongElec[12];
    TH1I* hNPEShortElec[12];
    TH1I* hNPELongPion[12];
    TH1I* hNPEShortPion[12];

};
