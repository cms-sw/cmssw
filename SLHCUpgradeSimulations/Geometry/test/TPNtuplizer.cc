// File: TPNtuplizer.cc
// Description: see TPNtuplizer.h
// Authors: H. Cheung
//--------------------------------------------------------------


#include "SLHCUpgradeSimulations/Geometry/test/TPNtuplizer.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// DataFormats
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

// For ROOT
#include <TROOT.h>
#include <TTree.h>
#include <TFile.h>
#include <TF1.h>
#include <TH2F.h>
#include <TH1F.h>

// STD
#include <memory>
#include <string>
#include <iostream>

using namespace std;
using namespace edm;
using namespace reco;

TPNtuplizer::TPNtuplizer(edm::ParameterSet const& conf) : 
  conf_(conf), 
  label_(conf.getParameter< std::vector<edm::InputTag> >("label")),
  label_tp_effic_(conf.getParameter< edm::InputTag >("label_tp_effic")),
  label_tp_fake_(conf.getParameter< edm::InputTag >("label_tp_fake")),
  UseAssociators_(conf.getParameter< bool >("UseAssociators")),
  associators_(conf.getParameter< std::vector<std::string> >("associators")),
  tfile_(0), 
  tptree_(0)
{
  tpSelector_ = TrackingParticleSelector(conf_.getParameter<double>("ptMinTP"),
                                         conf_.getParameter<double>("minRapidityTP"),
                                         conf_.getParameter<double>("maxRapidityTP"),
                                         conf_.getParameter<double>("tipTP"),
                                         conf_.getParameter<double>("lipTP"),
                                         conf_.getParameter<int>("minHitTP"),
                                         conf_.getParameter<bool>("signalOnlyTP"),
                                         conf_.getParameter<bool>("intimeOnlyTP"),
                                         conf_.getParameter<bool>("chargedOnlyTP"),
                                         conf_.getParameter<bool>("stableOnlyTP"),
                                         conf_.getParameter<std::vector<int> >("pdgIdTP"));
}

TPNtuplizer::~TPNtuplizer() { }  

void TPNtuplizer::endJob() 
{
  std::cout << " TPNtuplizer::endJob" << std::endl;
  tfile_->Write();
  tfile_->Close();
}

void TPNtuplizer::beginRun(Run const&, EventSetup const& es)
{
  std::string outputFile = conf_.getParameter<std::string>("OutputFile");
 
  tfile_ = new TFile ( outputFile.c_str() , "RECREATE" );
  tptree_ = new TTree("TPNtuple","Tracking Particle analyzer ntuple");

  int bufsize = 64000;

  //Common Branch
  tptree_->Branch("evt", &evt_, "run/I:evtnum:numtp:nseltp:nfdtp:numtk:nasstk", bufsize);
  tptree_->Branch("tpart", &tp_, 
    "tpn/I:bcross:tevt:charge:stable:status:pdgid:mathit:signal:llived:sel:gpsz:gpstat:pt/F:eta:tip:lip:p:e:phi:theta:rap:qual", bufsize);
  
}

// Functions that gets called by framework every event
void TPNtuplizer::analyze(const edm::Event& event, const edm::EventSetup& es)
{
  using namespace reco;

  edm::Handle<TrackingParticleCollection>  TPCollectionHeff ;
  event.getByLabel(label_tp_effic_,TPCollectionHeff);
  const TrackingParticleCollection tPCeff = *(TPCollectionHeff.product());
  
  edm::Handle<TrackingParticleCollection>  TPCollectionHfake ;
  event.getByLabel(label_tp_fake_,TPCollectionHfake);
  const TrackingParticleCollection tPCfake = *(TPCollectionHfake.product());

  if (UseAssociators_) {
    edm::Handle<reco::TrackToTrackingParticleAssociator> theAssociator;
    for (unsigned int w=0;w<associators_.size();w++) {
      event.getByLabel(associators_[w],theAssociator);
      associator_.push_back( theAssociator.product() );
    }
  }

  for (unsigned int ww=0;ww<associators_.size();ww++){
    // get some numbers for this event - very inefficient!
    int num_sel=0;
    for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
      TrackingParticleRef tpr(TPCollectionHeff, i);
      TrackingParticle* tp=const_cast<TrackingParticle*>(tpr.get());
      if( (! tpSelector_(*tp))) continue;
      ++num_sel;
    }
    edm::LogVerbatim("TPNtuplizer") << "\n# Tracking particles selected = " << num_sel << "\n";
    for (unsigned int www=0;www<label_.size();www++){
      // get track collection from event for specified collection label(s)
      edm::Handle<View<Track> >  trackCollection;
      event.getByLabel(label_[www], trackCollection);
      edm::LogVerbatim("TPNtuplizer") << "\n# of Reco tracks collection = " << trackCollection->size() << "\n";
      // do the association
      reco::RecoToSimCollection recSimColl;
      reco::SimToRecoCollection simRecColl;
      // only handle doing association in job at the mo
      if(UseAssociators_){
	recSimColl=associator_[ww]->associateRecoToSim(trackCollection, TPCollectionHfake);
         simRecColl=associator_[ww]->associateSimToReco(trackCollection, TPCollectionHeff);
      }
      // get number for this event and this track collection - very inefficient!
      int num_found=0;
      for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
        TrackingParticleRef tpr(TPCollectionHeff, i);
        TrackingParticle* tp=const_cast<TrackingParticle*>(tpr.get());
        if( (! tpSelector_(*tp))) continue;
        if(simRecColl.find(tpr) != simRecColl.end()) ++num_found;
      }
      int num_ass=0;
      for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){
        RefToBase<Track> track(trackCollection, i);
        std::vector<std::pair<TrackingParticleRef, double> > tp;
        if(recSimColl.find(track) != recSimColl.end()){
          tp = recSimColl[track];
          if (tp.size()!=0) ++num_ass;
        }
      } 
      edm::LogVerbatim("TPNtuplizer") << "\n# Tracking particles selected and found = " << num_found << "\n";
      edm::LogVerbatim("TPNtuplizer") << "\n# Reco tracks associated = " << num_ass << "\n";

      edm::LogVerbatim("TPNtuplizer") << "\n# of TrackingParticles: " << tPCeff.size() << "\n";
      edm::LogVerbatim("TPNtuplizer") << "\n# of Reco tracks for ntuple: " << trackCollection->size() << "\n";
      int ats = 0;
      int st=0;
      for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
        TrackingParticleRef tpr(TPCollectionHeff, i);
        TrackingParticle* tp=const_cast<TrackingParticle*>(tpr.get());
        //if( (! tpSelector_(*tp))) continue;
        int selected = 0;
        if( tpSelector_(*tp)) selected = 1;
        st++;

        std::vector<std::pair<RefToBase<Track>, double> > rt;
        float quality = 0.0;
        int matched_hit = 0;
        if(simRecColl.find(tpr) != simRecColl.end()){
          rt = (std::vector<std::pair<RefToBase<Track>, double> >) simRecColl[tpr];
          if (rt.size()!=0) {
            ats++;
            edm::LogVerbatim("TPNtuplizer") << "TrackingParticle #" << i << " selected #" << st 
                                            << " with pt=" << sqrt(tp->momentum().perp2()) 
                                            << " associated with quality:" << rt.begin()->second <<"\n";
            quality = rt.begin()->second;
            matched_hit = 1;
          }
        }else{
          edm::LogVerbatim("TPNtuplizer") << "TrackingParticle #" << i << " selected #" << st
                                          << " with pt=" << sqrt(tp->momentum().perp2())
                                          << " NOT associated to any reco::Track" << "\n";
        }
        fillEvt(tPCeff.size(), num_sel, num_found, trackCollection->size(), num_ass, event);
        fillTP(st, matched_hit, quality, selected, tp);
        tptree_->Fill();
        init();
      } // end loop over tracking particles

      // next reconstructed tracks
      edm::LogVerbatim("TPNtuplizer") << "\n# of reco::Tracks = " << trackCollection->size() << "\n";
      int at=0;
      int rT=0;
      for(View<Track>::size_type i=0; i<trackCollection->size(); ++i){
        RefToBase<Track> track(trackCollection, i);
        rT++;

        std::vector<std::pair<TrackingParticleRef, double> > tp;
        if(recSimColl.find(track) != recSimColl.end()){
          tp = recSimColl[track];
          if (tp.size()!=0) {
            at++;
            edm::LogVerbatim("TPNtuplizer") << "reco::Track #" << rT << " with pt=" << track->pt() 
                                            << " associated with quality:" << tp.begin()->second <<"\n";
          }
        } else {
          edm::LogVerbatim("TPNtuplizer") << "reco::Track #" << rT << " with pt=" << track->pt()
                                          << " NOT associated to any TrackingParticle" << "\n";		  
        }
      } // end loop over reco tracks

    } // end loop on track collection label
  } // end of loop on associators

} // end analyze function

void TPNtuplizer::fillTP(const int num, const int matched_hit, const float quality, 
                         const int selected, const TrackingParticle* tp)
{
  edm::LogVerbatim("TPNtuplizer") << "Filling TP with pt= " << sqrt(tp->momentum().perp2()) << "\n";

  tp_.tpn = num;
  tp_.bcross = tp->eventId().bunchCrossing();
  tp_.tevt = tp->eventId().event();
  tp_.charge = tp->charge();
  int stable = 1;
  for( TrackingParticle::genp_iterator j = tp->genParticle_begin(); j != tp->genParticle_end(); ++j ) {
    const reco::GenParticle * p = j->get();
    if (p->status() != 1) {
      stable = 0; break;
    }
  }
  tp_.stable = stable;
  tp_.status = tp->status();
  tp_.pdgid = tp->pdgId();
  tp_.mathit = matched_hit;
  if (tp->eventId().bunchCrossing()== 0 && tp->eventId().event() == 0) tp_.signal = 1;
    else tp_.signal = 0;
  if(tp->longLived()) tp_.llived = 1;
    else tp_.llived = 0;
  tp_.sel = selected;
  int numgp = 0;
  for( TrackingParticle::genp_iterator j = tp->genParticle_begin(); j != tp->genParticle_end(); ++ j ) ++numgp;
  int gpstatus = -69;
  for( TrackingParticle::genp_iterator j = tp->genParticle_begin(); j != tp->genParticle_end(); ++ j ) {
    const reco::GenParticle * p = j->get();
    if (p->status() != 1) {
      gpstatus = p->status(); break;
    }
    gpstatus = p->status();
  }
  tp_.gpsz = numgp;
  tp_.gpstat = gpstatus;
  tp_.pt = sqrt(tp->momentum().perp2());
  tp_.eta = tp->eta();
  tp_.tip = sqrt(tp->vertex().perp2());
  tp_.lip = fabs(tp->vertex().z());
  tp_.p = tp->p();
  tp_.e = tp->energy();
  tp_.phi = tp->phi();
  tp_.theta = tp->theta();
  tp_.rap = tp->rapidity();
  tp_.qual = quality;
}

void TPNtuplizer::fillEvt(const int numtp, const int nseltp, const int nfdtp,
                          const int numtk, const int nasstk, const edm::Event& E)
{
   evt_.run = E.id().run();
   evt_.evtnum = E.id().event();
   evt_.numtp = numtp;
   evt_.nseltp = nseltp;
   evt_.nfdtp = nfdtp;
   evt_.numtk = numtk;
   evt_.nasstk = nasstk;
}

void TPNtuplizer::init()
{
  evt_.init();
  tp_.init();
}

void TPNtuplizer::evt::init()
{
  int dummy_int = 9999;
  run = dummy_int;
  evtnum = dummy_int;
  numtp = dummy_int;
  numtk = dummy_int;
}

void TPNtuplizer::myTp::init()
{
  int dummy_int = 9999;
  float dummy_float = 9999.0;
  pt  = dummy_float;
  tpn = dummy_int;
  bcross = dummy_int;
  tevt = dummy_int;
  charge = dummy_int;
  stable = dummy_int;
  status = dummy_int;
  pdgid = dummy_int;
  mathit = dummy_int;
  signal = dummy_int;
  llived = dummy_int;
  sel = dummy_int;
  gpsz = dummy_int;
  gpstat = dummy_int;
  pt = dummy_float;
  eta = dummy_float;
  tip = dummy_float;
  lip = dummy_float;
  p = dummy_float;
  e = dummy_float;
  phi = dummy_float;
  theta = dummy_float;
  rap = dummy_float;
  qual = dummy_float;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TPNtuplizer);

