// -*- C++ -*-
//
// Package:    RPCPhiEff
// Class:      RPCPhiEff
//
/**\class RPCPhiEff RPCPhiEff.cc MyLib/RPCPhiEff/src/RPCPhiEff.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Tomasz Fruboes
//         Created:  Wed Mar  7 08:31:57 CET 2007
//
//


#include "Validation/MuonRPCGeometry/plugins/RPCPhiEff.h"
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <L1Trigger/RPCTrigger/interface/RPCLogCone.h>
#include <L1Trigger/RPCTrigger/interface/RPCConst.h>

#include "DataFormats/Math/interface/LorentzVector.h"

#include <iostream>
#include <set>
#include <fstream>
#include <sstream>

#include "L1Trigger/RPCTrigger/interface/RPCConst.h"

#include "DataFormats/Math/interface/deltaR.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
RPCPhiEff::RPCPhiEff(const edm::ParameterSet & ps) :
    m_rpcbToken( consumes<std::vector<L1MuRegionalCand> > (
        ps.getParameter< edm::InputTag >("rpcb"))),
    m_rpcfToken( consumes<std::vector<L1MuRegionalCand> > (
        ps.getParameter< edm::InputTag >("rpcf"))),
    m_g4Token( consumes<edm::SimTrackContainer>(
        ps.getParameter< edm::InputTag >("g4"))),
    m_rpcdigiToken( consumes<RPCDigiCollection> (
        ps.getParameter< edm::InputTag >("rpcdigi")))
{


  //m_outfileC.open("phieffC.txt");
  m_outfileR.open("muons.txt");

}


RPCPhiEff::~RPCPhiEff()
{

  //m_outfileC.close();
  m_outfileR.close();

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
RPCPhiEff::analyze(const edm::Event & iEvent,
                   const edm::EventSetup & iSetup)
{


  edm::Handle<std::vector<L1MuRegionalCand> > rpcBarrel;
  edm::Handle<std::vector<L1MuRegionalCand> > rpcForward;

  iEvent.getByToken(m_rpcbToken, rpcBarrel);
  iEvent.getByToken(m_rpcfToken, rpcForward);


  std::vector< edm::Handle<std::vector<L1MuRegionalCand> >  > handleVec;
  handleVec.push_back( rpcBarrel );
  handleVec.push_back( rpcForward );

  // Fetch MCdata
  edm::Handle<edm::SimTrackContainer> simTracks;
  iEvent.getByToken(m_g4Token, simTracks);

  float etaGen = 0, phiGen = 0, ptGen = 0;

  int noOfMuons = 0;
  int noOfRecMuons = 0;
  int noOfMatchedRecMuons = 0;

  bool firstRunForMuonMatchingCnt = true;

  // ask MC muons to be separated
  double deltarMin = -1;
  edm::SimTrackContainer::const_iterator simTrk = simTracks->begin();
  for (; simTrk != simTracks->end(); ++simTrk) {

    if (simTrk->type() != -13 && simTrk->type()!=13) continue;

    edm::SimTrackContainer::const_iterator simTrk2 = simTrk;
    ++simTrk2;
    for (; simTrk2 != simTracks->end(); ++simTrk2) {
      if (simTrk2->type() != -13 && simTrk2->type()!=13) continue;
      double drCand = reco::deltaR(simTrk2->momentum(), simTrk->momentum());
      if (drCand < deltarMin || deltarMin < 0) deltarMin = drCand;
    }


  }

  //std::cout << deltarMin << std::endl;
  if (deltarMin < 0.7 &&  deltarMin > 0) return;

  simTrk = simTracks->begin();
  for (; simTrk != simTracks->end(); simTrk++) {
    int type = simTrk->type();
    if (type == 13 || type == -13) {
      // Get the data
      const math::XYZTLorentzVectorD momentum = simTrk->momentum();
      etaGen = momentum.eta();
      ptGen = momentum.Pt();
      phiGen = momentum.phi();
      noOfMuons++;

      bool matched = false;
      int ptCodeRec = 0;
      int towerRec = 0;
      int phiRec = 0;
      //int muonsFound=0;
      int qual = 0;
      int ghost = 0; // number of ghost for montecarlo muon

      // Iter rpc muon cands, perform delta R matching
      // todo perform matching also using eta...
      for (  std::vector< edm::Handle<std::vector<L1MuRegionalCand> >  >::iterator it = handleVec.begin();
             it != handleVec.end();
             ++it  )
      {
        std::vector<L1MuRegionalCand>::const_iterator itRPC;
        for (itRPC=(*it)->begin(); itRPC!=(*it)->end(); itRPC++){
          int ptCode =  itRPC->pt_packed();
          if (ptCode != 0) {

            if (firstRunForMuonMatchingCnt) ++noOfRecMuons;
            ptCodeRec=ptCode;
            phiRec=itRPC->phi_packed();
            qual = itRPC->quality();
            towerRec = itRPC->eta_packed();
            if (towerRec > 16) {
              towerRec = - ( (~towerRec & 63) + 1);
            }
            // match
            float pi = 3.14159265;
            //float phiPhys = 2*pi*float(phiRec)/144-pi;
            float phiScaled = phiGen;
            if (phiScaled<0)  phiScaled += 2*pi;
            int phiHwGen = (phiScaled)/2/pi*144;

            //std::cout << "phi " << phiGen << " " << phiHwGen << " " << phiRec
            //          << " eta " << etaGen << " " << m_const.towerNumFromEta(etaGen) << " "<< towerRec << std::endl;
            if (  ( std::abs(phiHwGen-phiRec) < 10) && (std::abs(m_const.towerNumFromEta(etaGen)-towerRec)<1) )
            {
              if (matched) { // we have matched m.c. earlier, this is ghost
                ++ghost;
              }
              matched = true;
              ++noOfMatchedRecMuons;
              m_outfileR << etaGen << " " << phiGen << " " << ptGen
                         << " "    << towerRec << " " << phiRec << " " << ptCodeRec << " " << qual
                         << " "    << ghost
                         << std::endl;
            }
          } // (ptCode != 0)
        } // muon cands iter ends
      } // barrell/fwd iter ends
      firstRunForMuonMatchingCnt=false;
      if (!matched) {
        m_outfileR << etaGen << " " << phiGen << " " << ptGen
                   << " "    << 0 << " " << 0 << " " << 0 << " " << 0
                   << " "    << 0
                   << std::endl;

      }


    }
  }

  edm::EventID id = iEvent.id();
  edm::EventNumber_t evNum = id.event();
  edm::RunNumber_t rnNum = id.run();

  if (noOfMatchedRecMuons!=noOfRecMuons)  {
    edm::LogInfo("RPCEffWarn") << " MuonCands " << noOfRecMuons
                               << " matched " << noOfMatchedRecMuons
                               << " in run " << rnNum
                               << " event " << evNum;
  }



  /*
    m_outfileC << etaGen << " " << phiGen << " " << ptGen << " "
    << phiRec << " " << towerRec << " " << muonsFound  << " "
    <<  fromCones(iEvent) <<std::endl;*/
  /*m_outfileR << etaGen << " " << phiGen << " " << ptGen << " "
    << phiRec << " " << towerRec << " " << muonsFound  << " "
    <<  fromRaw(iEvent) <<std::endl;*/


  /*
    m_outfileR << etaGen << " " << phiGen << " " << ptGen << " "
    << phiRec << " " << towerRec << " " << muonsFound  << " "
    << std::endl;

  */


}

std::string RPCPhiEff::fromCones(const edm::Event & iEvent){

  return "";
}

// ------------ Check hw planes fired using rpc digis
std::string RPCPhiEff::fromRaw(const edm::Event & iEvent){

  std::stringstream ss;

  // Digi data.
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByToken(m_rpcdigiToken, rpcDigis);

  std::set<int> hwPlanes;

  RPCDigiCollection::DigiRangeIterator detUnitIt;
  for (detUnitIt=rpcDigis->begin();
       detUnitIt!=rpcDigis->end();
       ++detUnitIt)
  {

    const RPCDigiCollection::Range& range = (*detUnitIt).second;
    bool hasBX0 = false;
    for (RPCDigiCollection::const_iterator digiIt = range.first;
         digiIt!=range.second;
         ++digiIt)
    {
      if (digiIt->bx() == 0) {
        hasBX0 = true;
        break;
      }
    }

    if (!hasBX0) continue;

    const RPCDetId& id = (*detUnitIt).first;
    int station = id.station();
    int layer = id.layer();
    //int region = id.region();

    if (station == 3)
      hwPlanes.insert(5);

    else if (station == 4)
      hwPlanes.insert(6);

    else if (station  == 1 && layer == 1)
      hwPlanes.insert(1);

    else if (station  == 1 && layer == 2)
      hwPlanes.insert(2);

    else if (station  == 2 && layer == 1)
      hwPlanes.insert(3);

    else if (station  == 2 && layer == 2)
      hwPlanes.insert(4);

    else
      std::cout << "??????????????" << std::endl;

  }


  for (std::set<int>::iterator it= hwPlanes.begin();
       it!= hwPlanes.end();
       ++it)
  {
    ss << " " << *it;
  }




  return ss.str();



}

// ------------ method called once each job just before starting event loop  ------------
void RPCPhiEff::beginJob(const edm::EventSetup &)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void RPCPhiEff::endJob()
{
}

