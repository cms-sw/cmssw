#ifndef MuonRPCGeometry_RPCPhiEff_h
#define MuonRPCGeometry_RPCPhiEff_h

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <SimDataFormats/Track/interface/SimTrackContainer.h>

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

#include <L1Trigger/RPCTrigger/interface/RPCLogCone.h>
#include <L1Trigger/RPCTrigger/interface/RPCConst.h>

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"

#include "DataFormats/Math/interface/LorentzVector.h"

#include <iostream>
#include <set>
#include <fstream>
#include <sstream>

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

//
// class decleration
//

class RPCPhiEff:public edm::EDAnalyzer {
  public:
    explicit RPCPhiEff(const edm::ParameterSet &);
    ~RPCPhiEff();


  private:
    virtual void beginJob(const edm::EventSetup &);
    virtual void analyze(const edm::Event &, const edm::EventSetup &);
    std::string fromCones(const edm::Event & iEvent);
    std::string fromRaw(const edm::Event & iEvent);
    virtual void endJob();
    std::ofstream m_outfileC;
    std::ofstream m_outfileR;

    // ----------member data ---------------------------


    RPCConst rpcconst;
    //L1MuTriggerScales m_scales;
    edm::InputTag m_rpcb;
    edm::InputTag m_rpcf;
    edm::InputTag m_g4;
    edm::InputTag m_rpcdigi;

    RPCConst m_const;


};

#endif

