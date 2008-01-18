// -*- C++ -*-
//
// Package:    HSCParticleProducer
// Class:      HSCParticleProducer
// 
/**\class HSCParticleProducer HSCParticleProducer.cc SUSYBSMAnalysis/HSCParticleProducer/src/HSCParticleProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Rizzi Andrea
//         Created:  Wed Oct 10 12:01:28 CEST 2007
// $Id: HSCParticleProducer.cc,v 1.2 2007/11/21 13:18:06 arizzi Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "RecoTracker/DeDx/interface/DeDxEstimatorProducer.h"
#include "DataFormats/TrackReco/interface/TrackDeDxEstimate.h"
#include "DataFormats/TrackReco/interface/TrackDeDxHits.h"
#include "DataFormats/TrackReco/interface/DeDxHit.h"
#include "DataFormats/TrackReco/interface/Track.h"


#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCParticle.h"
#include "Math/GenVector/VectorUtil.h"


#include <vector>
#include <TNtuple.h>
#include <TF1.h>
#include <iostream>
//
// class decleration
//
using namespace susybsm;
class HSCParticleProducer : public edm::EDProducer {
   public:
      explicit HSCParticleProducer(const edm::ParameterSet&);
      ~HSCParticleProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      edm::InputTag m_trackDeDxHitsTag;
      edm::InputTag m_trackDeDxFitTag;
      edm::InputTag m_trackDeDxEstimatorTag;
      edm::InputTag m_muonsTag;
      edm::InputTag m_muonsTOFTag;

      TNtuple * tmpNt;
      TF1 * f1;
      // ----------member data ---------------------------
      typedef std::vector<DeDxBeta> DeDxBetaCollection; 
      std::vector<HSCParticle> associate( DeDxBetaCollection & tk ,const MuonTOFCollection & dts );


};

HSCParticleProducer::HSCParticleProducer(const edm::ParameterSet& iConfig)
{
using namespace edm;
using namespace std;
   m_trackDeDxHitsTag = iConfig.getParameter<edm::InputTag>("trackDeDxHits");
   m_trackDeDxFitTag = iConfig.getParameter<edm::InputTag>("trackDeDxFit");
   m_trackDeDxEstimatorTag = iConfig.getParameter<edm::InputTag>("trackDeDxEstimator");
   m_muonsTag = iConfig.getParameter<edm::InputTag>("muons");
   m_muonsTOFTag = iConfig.getParameter<edm::InputTag>("muonsTOF");

   produces<susybsm::HSCParticleCollection >();




}


HSCParticleProducer::~HSCParticleProducer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
HSCParticleProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

using namespace edm;
using namespace reco;
using namespace std;
using namespace susybsm;



/*   Handle<vector<float> >  betaRecoH;
   iEvent.getByLabel(m_muonsTOFTag,betaRecoH); // "betaFromTOF"
   const vector<float> & betaReco = *betaRecoH.product();
   Handle<reco::MuonCollection> muonsH;
   iEvent.getByLabel(m_muonsTag,muonsH); //"muons"
   const reco::MuonCollection & muons = * muonsH.product();
   MuonRefProd mr = MuonRefProd(muonsH);
   MuonTOFCollection dtInfos( mr );
   int i=0;
   reco::MuonCollection::const_iterator muonIt = muons.begin();
   for(; muonIt != muons.end() ; ++muonIt)
    {
      double invbeta = betaReco.at(i);
      DriftTubeTOF dt;
      dt.invBeta = invbeta;
      dtInfos.setValue(i,dt);
      i++;
    }
*/
  Handle<MuonTOFCollection> betaRecoH;
  iEvent.getByLabel(m_muonsTOFTag,betaRecoH);
  const MuonTOFCollection & dtInfos  =  *betaRecoH.product();



   Handle<TrackDeDxEstimateCollection> dedxH;
   Handle<TrackDeDxHitsCollection> dedxHitsH;
   Handle<vector<float> >  dedxFitH;
//   iEvent.getByLabel("dedxTruncated40",dedxH);
   iEvent.getByLabel(m_trackDeDxHitsTag,dedxHitsH);
   iEvent.getByLabel(m_trackDeDxEstimatorTag,dedxH);
   iEvent.getByLabel(m_trackDeDxFitTag,dedxFitH);

   const TrackDeDxEstimateCollection & dedx = *dedxH.product();
   const TrackDeDxHitsCollection & dedxHits = *dedxHitsH.product();

   const vector<float> & dedxFit = *dedxFitH.product();
   DeDxBetaCollection   tkInfos;
   for(size_t i=0; i<dedx.size() ; i++)
    {
        int usedhits=0;
        for(reco::DeDxHitCollection::const_iterator it_hits = dedxHits[i].second.begin(); it_hits!=dedxHits[i].second.end();it_hits++) 
         {  if(it_hits->subDet() != 1 && it_hits->subDet() != 2 ) usedhits++;       }

       if(dedx[i].first->normalizedChi2() < 5 && dedx[i].first->numberOfValidHits()>8 && usedhits >= 9)
       {
        float dedxVal= dedx[i].second;
        float dedxFitVal= dedxFit[i];
        float k=0.4;  //919/2.75*0.0012;
        float k2=0.432; //919/2.55*0.0012;
	DeDxBeta tk;
	tk.track=dedx[i].first;
        tk.invBeta2 = k*dedxVal;
        tk.invBeta2Fit = k2*dedxFitVal;
        //tk.nDeDxHits = dedxHits[i].second.size();
        tk.nDeDxHits = usedhits;
 
        tkInfos.push_back(tk);
        }
    }


    
   
   susybsm::HSCParticleCollection * hscp = new susybsm::HSCParticleCollection; 
   std::auto_ptr<susybsm::HSCParticleCollection> result(hscp);
   *hscp=associate(tkInfos,dtInfos);

   iEvent.put(result); 
}

// ------------ method called once each job just before starting event loop  ------------
void 
HSCParticleProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
HSCParticleProducer::endJob() {
}



std::vector<HSCParticle> HSCParticleProducer::associate( DeDxBetaCollection & tks ,const MuonTOFCollection & dtsInput )
{
 float minTkP=30;
 float maxTkBeta=0.9;

 float minDtP=30;

 float minDR=0.1;
 float maxInvPtDiff=0.005;

 float minTkInvBeta2=1./(maxTkBeta*maxTkBeta);
 std::vector<MuonTOF> dts;
 std::copy (dtsInput.begin(),dtsInput.end(), std::back_inserter (dts));

 std::vector<HSCParticle> result;
 for(size_t i=0;i<tks.size();i++)
 {
   if( tks[i].track.isNonnull() && tks[i].track->pt() > minTkP && tks[i].invBeta2 >= minTkInvBeta2 )
    {
       float min=1000;
       int found = -1;
       for(size_t j=0;j<dts.size();j++)
        {
         if(dts[j].first->combinedMuon().isNonnull())
          {
          float invDT=1./dts[j].first->combinedMuon()->pt();
          float invTK=1./tks[i].track->pt();
          if(fabs(invDT-invTK) > maxInvPtDiff) continue;
          float deltaR=ROOT::Math::VectorUtil::DeltaR(dts[j].first->combinedMuon()->momentum(), tks[i].track->momentum());
          if(deltaR > minDR || deltaR > min) continue;
          min=deltaR;
          found = j;
          }
       }
     HSCParticle candidate;
     candidate.tk=tks[i];
     candidate.hasDt=false;
     if(found>=0 )
       {
        candidate.hasDt=true;
        candidate.dt=dts[found];
        dts.erase(dts.begin()+found);
       }
      else
        {
//          if( tks[i].invBeta2 >= 1.30)
          std::cout << "Not found for " << tks[i].track->momentum() << " " << tks[i].track->eta() << std::endl;
        }
     result.push_back(candidate);

    }
 }

 for(size_t i=0;i<dts.size();i++)
 {
     if(dts[i].first->combinedMuon().isNonnull() && dts[i].first->combinedMuon()->pt() > minDtP  )
    {
       float min=1000;
       int found = -1;
       for(size_t j=0;j<tks.size();j++)
        {
         if( tks[j].track.isNonnull() )
         {
          float invDT=1./dts[i].first->combinedMuon()->pt();
          float invTK=1./tks[j].track->pt();
          if(fabs(invDT-invTK) > maxInvPtDiff) continue;
          float deltaR=ROOT::Math::VectorUtil::DeltaR(dts[i].first->combinedMuon()->momentum(), tks[j].track->momentum());
          if(deltaR > minDR || deltaR > min) continue;
          min=deltaR;
          found = j;
          std::cout << "At least two muons associated to the same track ?" << std::endl;
         }
       }
     HSCParticle candidate;
     candidate.dt=dts[i];
     candidate.hasTk=false;
     if(found>=0 )
       {
        candidate.hasTk=true;
        candidate.tk=tks[found];
       // tks.erase(tks.begin()+found);
       }
     result.push_back(candidate);
    }



 }
 std::cout << "return" << std::endl;
 return result;

}


//define this as a plug-in
DEFINE_FWK_MODULE(HSCParticleProducer);
