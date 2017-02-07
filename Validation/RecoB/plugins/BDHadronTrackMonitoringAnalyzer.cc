#include "Validation/RecoB/plugins/BDHadronTrackMonitoringAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h" 

#include "DQMOffline/RecoB/interface/Tools.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "RecoVertex/VertexPrimitives/interface/ConvertToFromReco.h"

#include <iostream>
#include <fstream>

using namespace reco;
using namespace edm;
using namespace std;


std::map<unsigned int, std::string> TrkHistCat{
    {0, "BCWeakDecay"},
    {1, "BWeakDecay"},
    {2, "CWeakDecay"},
    {3, "PU"},
    {4, "Other"},
    {5, "Fake"} 
};

const reco::TrackBaseRef toTrackRef(const edm::Ptr<reco::Candidate> & cnd)
{
    const reco::PFCandidate * pfcand = dynamic_cast<const reco::PFCandidate *>(cnd.get());

    if ( (std::abs(pfcand->pdgId()) == 11 || pfcand->pdgId() == 22) && pfcand->gsfTrackRef().isNonnull() && pfcand->gsfTrackRef().isAvailable() )
      return reco::TrackBaseRef(pfcand->gsfTrackRef());
    else if ( pfcand->trackRef().isNonnull() && pfcand->trackRef().isAvailable() )
      return reco::TrackBaseRef(pfcand->trackRef());
    else
      return reco::TrackBaseRef();
}


// ---------- Constructor -----------

BDHadronTrackMonitoringAnalyzer::BDHadronTrackMonitoringAnalyzer(const edm::ParameterSet& pSet) :
    distJetAxis_ ( pSet.getParameter<double>("distJetAxisCut") ),
    decayLength_ ( pSet.getParameter<double>("decayLengthCut") ),
    minJetPt_ ( pSet.getParameter<double>("minJetPt") ),
    maxJetEta_ ( pSet.getParameter<double>("maxJetEta") ),
    ipTagInfos_ ( pSet.getParameter<std::string>("ipTagInfos") ),
    PatJetSrc_ ( pSet.getParameter<InputTag>("PatJetSource") ),
    TrackSrc_ ( pSet.getParameter<InputTag>("TrackSource") ),
    PVSrc_ ( pSet.getParameter<InputTag>("PrimaryVertexSource") ),
    ClusterTPMapSrc_ ( pSet.getParameter<InputTag>("clusterTPMap") ),
    classifier_(pSet, consumesCollector())

{
    PatJetCollectionTag_ = consumes<pat::JetCollection>(PatJetSrc_);
    TrackCollectionTag_ = consumes<reco::TrackCollection>(TrackSrc_);
    PrimaryVertexColl_ = consumes<reco::VertexCollection>(PVSrc_);
    clusterTPMapToken_ = consumes<ClusterTPAssociation>(ClusterTPMapSrc_);
}



// ---------- BookHistograms -----------

void BDHadronTrackMonitoringAnalyzer::bookHistograms(DQMStore::IBooker & ibook, edm::Run const & run, edm::EventSetup const & es)
{
  //
  // Book all histograms.
  //
  RecoBTag::setTDRStyle();


  nTrkAll_bjet = ibook.book1D("nTrkAll_bjet","Number of selected tracks in b jets;number of selected tracks;jets",16,-0.5,15.5);
  //nTrkTruthAll_bjet = ibook.book1D("nTrkTruthAll_bjet","Number of selected TrackingParticles in b jets;number of selected truth tracks;jets",16,-0.5,15.5);

  nTrkAll_cjet = ibook.book1D("nTrkAll_cjet","Number of selected tracks in c jets;number of selected tracks;jets",16,-0.5,15.5);
  //nTrkTruthAll_cjet = ibook.book1D("nTrkTruthAll_cjet","Number of selected TrackingParticles in c jets;number of selected truth tracks;jets",16,-0.5,15.5);

  nTrkAll_dusgjet = ibook.book1D("nTrkAll_dusgjet","Number of selected tracks in dusg jets;number of selected tracks;jets",16,-0.5,15.5);
  //nTrkTruthAll_dusgjet = ibook.book1D("nTrkTruthAll_dusgjet","Number of selected TrackingParticles in dusg jets;number of selected truth tracks;jets",16,-0.5,15.5);

  // Loop over different Track History Categories
  for (unsigned int i = 0; i < TrkHistCat.size(); i++){
    // b jets
    nTrk_bjet[i] = ibook.book1D("nTrk_bjet_"+TrkHistCat[i],"Number of selected tracks in b jets ("+TrkHistCat[i]+");number of selected tracks ("+TrkHistCat[i]+");jets",16,-0.5,15.5);
    //nTrkTruth_bjet[i] = ibook.book1D("nTrkTruth_bjet_"+TrkHistCat[i],"Number of selected trackingparticles in b jets ("+TrkHistCat[i]+" Truth);number of selected tracks ("+TrkHistCat[i]+" Truth);jets",16,-0.5,15.5);

    // c jets
    nTrk_cjet[i] = ibook.book1D("nTrk_cjet_"+TrkHistCat[i],"Number of selected tracks in c jets ("+TrkHistCat[i]+");number of selected tracks ("+TrkHistCat[i]+");jets",16,-0.5,15.5);
    //nTrkTruth_cjet[i] = ibook.book1D("nTrkTruth_cjet_"+TrkHistCat[i],"Number of selected trackingparticles in c jets ("+TrkHistCat[i]+" Truth);number of selected tracks ("+TrkHistCat[i]+" Truth);jets",16,-0.5,15.5);

    // dusg jets
    nTrk_dusgjet[i] = ibook.book1D("nTrk_dusgjet_"+TrkHistCat[i],"Number of selected tracks in dusg jets ("+TrkHistCat[i]+");number of selected tracks ("+TrkHistCat[i]+");jets",16,-0.5,15.5);
    //nTrkTruth_dusgjet[i] = ibook.book1D("nTrkTruth_dusgjet_"+TrkHistCat[i],"Number of selected trackingparticles in dusg jets ("+TrkHistCat[i]+" Truth);number of selected tracks ("+TrkHistCat[i]+" Truth);jets",16,-0.5,15.5);


    // track properties for all flavours combined
    TrkPt_alljets[i] = ibook.book1D("TrkPt_"+TrkHistCat[i],"Track pT ("+TrkHistCat[i]+");track p_{T} ("+TrkHistCat[i]+");tracks",30,0,100);
    TrkEta_alljets[i] = ibook.book1D("TrkEta_"+TrkHistCat[i],"Track #eta ("+TrkHistCat[i]+");track #eta ("+TrkHistCat[i]+");tracks",30,-2.5,2.5);
    TrkPhi_alljets[i] = ibook.book1D("TrkPhi_"+TrkHistCat[i],"Track #phi ("+TrkHistCat[i]+");track #phi ("+TrkHistCat[i]+");tracks",30,-3.15,3.15);
    TrkDxy_alljets[i] = ibook.book1D("TrkDxy_"+TrkHistCat[i],"Track dxy ("+TrkHistCat[i]+");track dxy ("+TrkHistCat[i]+");tracks",30,-0.1,0.1);
    TrkDz_alljets[i] = ibook.book1D("TrkDz_"+TrkHistCat[i],"Track dz ("+TrkHistCat[i]+");track dz ("+TrkHistCat[i]+");tracks",30,-0.1,0.1);
    TrkHitAll_alljets[i] = ibook.book1D("TrkHitAll_"+TrkHistCat[i],"Number of tracker hits ("+TrkHistCat[i]+");track number of all hits ("+TrkHistCat[i]+");tracks",31,-0.5,30.5);
    TrkHitStrip_alljets[i] = ibook.book1D("TrkHitStrip_"+TrkHistCat[i],"Number of strip hits ("+TrkHistCat[i]+");track number of strip hits ("+TrkHistCat[i]+");tracks",31,-0.5,30.5);
    TrkHitPixel_alljets[i] = ibook.book1D("TrkHitPixel_"+TrkHistCat[i],"Number of pixel hits ("+TrkHistCat[i]+");track number of pixel hits ("+TrkHistCat[i]+");tracks",9,-0.5,8.5);
    if (i < 5){ // Fakes (i == 5) have no truth by definition!
        TrkTruthPt_alljets[i] = ibook.book1D("TrkTruthPt_"+TrkHistCat[i],"Track pT ("+TrkHistCat[i]+" Truth);track p_{T} ("+TrkHistCat[i]+" Truth);tracks",30,0,100);
        TrkTruthEta_alljets[i] = ibook.book1D("TrkTruthEta_"+TrkHistCat[i],"Track #eta ("+TrkHistCat[i]+" Truth);track #eta ("+TrkHistCat[i]+" Truth);tracks",30,-2.5,2.5);
        TrkTruthPhi_alljets[i] = ibook.book1D("TrkTruthPhi_"+TrkHistCat[i],"Track #phi ("+TrkHistCat[i]+" Truth);track #phi ("+TrkHistCat[i]+" Truth);tracks",30,-3.15,3.15);
        TrkTruthDxy_alljets[i] = ibook.book1D("TrkTruthDxy_"+TrkHistCat[i],"Track dxy ("+TrkHistCat[i]+" Truth);track dxy ("+TrkHistCat[i]+" Truth);tracks",30,-0.1,0.1);
        TrkTruthDz_alljets[i] = ibook.book1D("TrkTruthDz_"+TrkHistCat[i],"Track dz ("+TrkHistCat[i]+" Truth);track dz ("+TrkHistCat[i]+" Truth);tracks",30,-0.1,0.1);
        TrkTruthHitAll_alljets[i] = ibook.book1D("TrkTruthHitAll_"+TrkHistCat[i],"Number of tracker hits ("+TrkHistCat[i]+" Truth);track number of all hits ("+TrkHistCat[i]+" Truth);tracks",31,-0.5,30.5);
        TrkTruthHitStrip_alljets[i] = ibook.book1D("TrkTruthHitStrip_"+TrkHistCat[i],"Number of strip hits ("+TrkHistCat[i]+" Truth);track number of strip hits ("+TrkHistCat[i]+" Truth);tracks",31,-0.5,30.5);
        TrkTruthHitPixel_alljets[i] = ibook.book1D("TrkTruthHitPixel_"+TrkHistCat[i],"Number of pixel hits ("+TrkHistCat[i]+" Truth);track number of pixel hits ("+TrkHistCat[i]+" Truth);tracks",9,-0.5,8.5);
    }
  }
}


// ---------- Destructor -----------

BDHadronTrackMonitoringAnalyzer::~BDHadronTrackMonitoringAnalyzer()
{  
}


// ---------- Analyze -----------
// This is needed to get a TrackingParticle --> Cluster match (instead of Cluster-->TP) 
using P = std::pair<OmniClusterRef, TrackingParticleRef>;
bool compare(const P& i, const P& j) {
    return i.second.index() > j.second.index();
}

void BDHadronTrackMonitoringAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  edm::Handle<pat::JetCollection> patJetsColl;
  iEvent.getByToken(PatJetCollectionTag_, patJetsColl);

  edm::Handle<reco::TrackCollection> tracksHandle;
  iEvent.getByToken(TrackCollectionTag_,tracksHandle);

  edm::Handle<ClusterTPAssociation> pCluster2TPListH;
  iEvent.getByToken(clusterTPMapToken_, pCluster2TPListH);
  const ClusterTPAssociation& clusterToTPMap = *pCluster2TPListH;

  edm::ESHandle<TransientTrackBuilder> trackBuilder ;
  iSetup.get<TransientTrackRecord>().get("TransientTrackBuilder", trackBuilder);

  classifier_.newEvent(iEvent, iSetup);

  // -----Primary Vertex-----
  const reco::Vertex *pv;

  edm::Handle<reco::VertexCollection> primaryVertex ;
  iEvent.getByToken(PrimaryVertexColl_,primaryVertex);

  bool pvFound = (primaryVertex->size() != 0);
  if ( pvFound ) {
    pv = &(*primaryVertex->begin());
  }
  else {
    reco::Vertex::Error e;
    e(0,0)=0.0015*0.0015;
    e(1,1)=0.0015*0.0015;
    e(2,2)=15.*15.;
    reco::Vertex::Point p(0,0,0);
    pv=  new reco::Vertex(p,e,1,1,1);
  }
  // -----------------------


  // -------- Loop Over Jets ----------
  for ( pat::JetCollection::const_iterator jet = patJetsColl->begin(); jet != patJetsColl->end(); ++jet ) {
    if ( ( jet->pt() < minJetPt_ || std::fabs( jet->eta() ) > maxJetEta_ ) ) continue;

    unsigned int flav = abs(jet->hadronFlavour());


    //std::cout << "is there a taginfo? " << jet->hasTagInfo(ipTagInfos_.c_str()) << std::endl;
    const CandIPTagInfo *trackIpTagInfo = jet->tagInfoCandIP(ipTagInfos_.c_str());
    const std::vector<edm::Ptr<reco::Candidate> > & selectedTracks( trackIpTagInfo->selectedTracks() );


    unsigned int nseltracks = 0;
    int nseltracksCat[6] = {0,0,0,0,0,0}; // following the order of TrkHistCat
    //int nseltracksTruthCat[6] = {0,0,0,0,0,0}; // following the order of TrkHistCat
    //unsigned int nseltracksTruth = 0;

    unsigned int nTrackSize = selectedTracks.size(); // number of tracks from IPInfos to loop over
    // -------- Loop Over (selected) Tracks ----------
    for (unsigned int itt=0; itt < nTrackSize; ++itt)
    {
        const TrackBaseRef ptrackRef = toTrackRef(selectedTracks[itt]);
        const reco::Track * ptrackPtr = reco::btag::toTrack(ptrackRef);
        const reco::Track & ptrack = *ptrackPtr;
    
        reco::TransientTrack transientTrack = trackBuilder->build(ptrackPtr);
        GlobalVector direction(jet->px(), jet->py(), jet->pz());
    
        Double_t distJetAxis = IPTools::jetTrackDistance(transientTrack, direction, *pv).second.value();
    
        Double_t decayLength=999;
        TrajectoryStateOnSurface closest = IPTools::closestApproachToJet(transientTrack.impactPointState(), *pv, direction, transientTrack.field());
        if (closest.isValid())
            decayLength =  (closest.globalPosition() - RecoVertex::convertPos(pv->position())).mag();
        else
            decayLength = 999;
    
        // extra cut ons the tracks
        if (std::fabs(distJetAxis) > distJetAxis_ || decayLength > decayLength_){
            continue;
        }
        nseltracks+=1; // if it passed these cuts, nselectedtracks +1
    
    
        TrackCategories::Flags theFlag = classifier_.evaluate( toTrackRef(selectedTracks[itt]) ).flags();
    
        double TrkPt = ptrack.pt();
        double TrkEta = ptrack.eta();
        double TrkPhi = ptrack.phi();
        double TrkDxy = ptrack.dxy(pv->position());
        double TrkDz = ptrack.dz(pv->position());
        int TrknHitAll = ptrack.numberOfValidHits();
        int TrknHitPixel = ptrack.hitPattern().numberOfValidPixelHits();
        int TrknHitStrip = ptrack.hitPattern().numberOfValidStripHits();

    
        double TrkTruthPt=-99;
        double TrkTruthEta=-99;
        double TrkTruthPhi=-99;
        double TrkTruthDxy=-1;
        double TrkTruthDz=-1;
        int TrkTruthnHitAll=-1;
        int TrkTruthnHitPixel=-1;
        int TrkTruthnHitStrip=-1;
    
        // Get corresponding Trackingparticle
        std::pair<TrackingParticleRef, double> res = classifier_.history().getMatchedTrackingParticle();
        TrackingParticleRef tpr = res.first;
        double quality_tpr = res.second;
    
        // Match TP to hit-cluster (re-ordering according to TP rather than clusters and look for equal_range of a given tpr)
        auto clusterTPmap = clusterToTPMap.map();
        std::sort(clusterTPmap.begin(), clusterTPmap.end(), compare);
        auto clusterRange = std::equal_range(clusterTPmap.begin(), clusterTPmap.end(),std::make_pair(OmniClusterRef(), tpr), compare);
        if (quality_tpr != 0) {
            //nseltracksTruth +=1;
        
            TrkTruthPt = tpr->pt();
            TrkTruthEta = tpr->eta();
            TrkTruthPhi = tpr->phi();
        
            TrackingParticle::Point vertex_pv = pv->position();
            TrackingParticle::Point vertex_tpr = tpr->vertex();
            TrackingParticle::Vector momentum_tpr = tpr->momentum();
            TrkTruthDxy = (-(vertex_tpr.x()-vertex_pv.x())*momentum_tpr.y()+(vertex_tpr.y()-vertex_pv.y())*momentum_tpr.x())/tpr->pt();
            TrkTruthDz = (vertex_tpr.z()-vertex_pv.z()) - ((vertex_tpr.x()-vertex_pv.x())*momentum_tpr.x()+(vertex_tpr.y()-vertex_pv.y())*momentum_tpr.y())/sqrt(momentum_tpr.perp2()) * momentum_tpr.z()/sqrt(momentum_tpr.perp2());

            TrkTruthnHitAll = 0;
            TrkTruthnHitPixel = 0;
            TrkTruthnHitStrip = 0;
            if( clusterRange.first != clusterRange.second ) {
                for( auto ip=clusterRange.first; ip != clusterRange.second; ++ip ) {
                    const OmniClusterRef& cluster = ip->first;
                    if (cluster.isPixel() && cluster.isValid()){ TrkTruthnHitPixel+=1;}
                    if (cluster.isStrip() && cluster.isValid()){ TrkTruthnHitStrip+=1;}
                }
            }
            TrkTruthnHitAll = TrkTruthnHitPixel + TrkTruthnHitStrip;
        
            /*
            if ( theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::BWeakDecay] && theFlag[TrackCategories::CWeakDecay] ) {nseltracksTruthCat[0] += 1;}
            else if ( theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::BWeakDecay] && !theFlag[TrackCategories::CWeakDecay] ) {nseltracksTruthCat[1] += 1;}
            else if ( theFlag[TrackCategories::SignalEvent] && !theFlag[TrackCategories::BWeakDecay] && theFlag[TrackCategories::CWeakDecay] ) {nseltracksTruthCat[2] += 1;}
            else if ( !theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::Fake] ) {nseltracksTruthCat[3] += 1;}
            else if ( !theFlag[TrackCategories::SignalEvent] && !theFlag[TrackCategories::Fake] ) {nseltracksTruthCat[4] += 1;}
            else{ nseltracksTruthCat[5] += 1; }
            */
        
        }
    
    
    
        // ----------- Filling the correct histograms based on jet flavour and Track history Category --------
    
    
        //BCWeakDecay
        if ( theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::BWeakDecay] && theFlag[TrackCategories::CWeakDecay] ) {
            nseltracksCat[0] += 1;
            TrkPt_alljets[0]->Fill(TrkPt);
            TrkEta_alljets[0]->Fill(TrkEta);
            TrkPhi_alljets[0]->Fill(TrkPhi);
            TrkDxy_alljets[0]->Fill(TrkDxy);
            TrkDz_alljets[0]->Fill(TrkDz);
            TrkHitAll_alljets[0]->Fill(TrknHitAll);
            TrkHitPixel_alljets[0]->Fill(TrknHitPixel);
            TrkHitStrip_alljets[0]->Fill(TrknHitStrip);
            if (quality_tpr != 0) {
                TrkTruthPt_alljets[0]->Fill(TrkTruthPt);
                TrkTruthEta_alljets[0]->Fill(TrkTruthEta);
                TrkTruthPhi_alljets[0]->Fill(TrkTruthPhi);
                TrkTruthDxy_alljets[0]->Fill(TrkTruthDxy);
                TrkTruthDz_alljets[0]->Fill(TrkTruthDz);
                TrkTruthHitAll_alljets[0]->Fill(TrkTruthnHitAll);
                TrkTruthHitPixel_alljets[0]->Fill(TrkTruthnHitPixel);
                TrkTruthHitStrip_alljets[0]->Fill(TrkTruthnHitStrip);
            }
        }
        //BWeakDecay
        else if ( theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::BWeakDecay] && !theFlag[TrackCategories::CWeakDecay] ) {
            nseltracksCat[1] += 1;
            TrkPt_alljets[1]->Fill(TrkPt);
            TrkEta_alljets[1]->Fill(TrkEta);
            TrkPhi_alljets[1]->Fill(TrkPhi);
            TrkDxy_alljets[1]->Fill(TrkDxy);
            TrkDz_alljets[1]->Fill(TrkDz);
            TrkHitAll_alljets[1]->Fill(TrknHitAll);
            TrkHitPixel_alljets[1]->Fill(TrknHitPixel);
            TrkHitStrip_alljets[1]->Fill(TrknHitStrip);
            if (quality_tpr != 0) {
                TrkTruthPt_alljets[1]->Fill(TrkTruthPt);
                TrkTruthEta_alljets[1]->Fill(TrkTruthEta);
                TrkTruthPhi_alljets[1]->Fill(TrkTruthPhi);
                TrkTruthDxy_alljets[1]->Fill(TrkTruthDxy);
                TrkTruthDz_alljets[1]->Fill(TrkTruthDz);
                TrkTruthHitAll_alljets[1]->Fill(TrkTruthnHitAll);
                TrkTruthHitPixel_alljets[1]->Fill(TrkTruthnHitPixel);
                TrkTruthHitStrip_alljets[1]->Fill(TrkTruthnHitStrip);
            }
        }
        //CWeakDecay
        else if ( theFlag[TrackCategories::SignalEvent] && !theFlag[TrackCategories::BWeakDecay] && theFlag[TrackCategories::CWeakDecay] ) {
            nseltracksCat[2] += 1;
            TrkPt_alljets[2]->Fill(TrkPt);
            TrkEta_alljets[2]->Fill(TrkEta);
            TrkPhi_alljets[2]->Fill(TrkPhi);
            TrkDxy_alljets[2]->Fill(TrkDxy);
            TrkDz_alljets[2]->Fill(TrkDz);
            TrkHitAll_alljets[2]->Fill(TrknHitAll);
            TrkHitPixel_alljets[2]->Fill(TrknHitPixel);
            TrkHitStrip_alljets[2]->Fill(TrknHitStrip);
            if (quality_tpr != 0) {
                TrkTruthPt_alljets[2]->Fill(TrkTruthPt);
                TrkTruthEta_alljets[2]->Fill(TrkTruthEta);
                TrkTruthPhi_alljets[2]->Fill(TrkTruthPhi);
                TrkTruthDxy_alljets[2]->Fill(TrkTruthDxy);
                TrkTruthDz_alljets[2]->Fill(TrkTruthDz);
                TrkTruthHitAll_alljets[2]->Fill(TrkTruthnHitAll);
                TrkTruthHitPixel_alljets[2]->Fill(TrkTruthnHitPixel);
                TrkTruthHitStrip_alljets[2]->Fill(TrkTruthnHitStrip);
            }
        }
        //PU
        else if ( !theFlag[TrackCategories::SignalEvent] && !theFlag[TrackCategories::Fake] ) {
            nseltracksCat[3] += 1;
            TrkPt_alljets[3]->Fill(TrkPt);
            TrkEta_alljets[3]->Fill(TrkEta);
            TrkPhi_alljets[3]->Fill(TrkPhi);
            TrkDxy_alljets[3]->Fill(TrkDxy);
            TrkDz_alljets[3]->Fill(TrkDz);
            TrkHitAll_alljets[3]->Fill(TrknHitAll);
            TrkHitPixel_alljets[3]->Fill(TrknHitPixel);
            TrkHitStrip_alljets[3]->Fill(TrknHitStrip);
            if (quality_tpr != 0) {
                TrkTruthPt_alljets[3]->Fill(TrkTruthPt);
                TrkTruthEta_alljets[3]->Fill(TrkTruthEta);
                TrkTruthPhi_alljets[3]->Fill(TrkTruthPhi);
                TrkTruthDxy_alljets[3]->Fill(TrkTruthDxy);
                TrkTruthDz_alljets[3]->Fill(TrkTruthDz);
                TrkTruthHitAll_alljets[3]->Fill(TrkTruthnHitAll);
                TrkTruthHitPixel_alljets[3]->Fill(TrkTruthnHitPixel);
                TrkTruthHitStrip_alljets[3]->Fill(TrkTruthnHitStrip);
            }
        }
        //Other
        else if ( theFlag[TrackCategories::SignalEvent] && !theFlag[TrackCategories::BWeakDecay] && !theFlag[TrackCategories::CWeakDecay] ){
            nseltracksCat[4] += 1;
            TrkPt_alljets[4]->Fill(TrkPt);
            TrkEta_alljets[4]->Fill(TrkEta);
            TrkPhi_alljets[4]->Fill(TrkPhi);
            TrkDxy_alljets[4]->Fill(TrkDxy);
            TrkDz_alljets[4]->Fill(TrkDz);
            TrkHitAll_alljets[4]->Fill(TrknHitAll);
            TrkHitPixel_alljets[4]->Fill(TrknHitPixel);
            TrkHitStrip_alljets[4]->Fill(TrknHitStrip);
            if (quality_tpr != 0) {
                TrkTruthPt_alljets[4]->Fill(TrkTruthPt);
                TrkTruthEta_alljets[4]->Fill(TrkTruthEta);
                TrkTruthPhi_alljets[4]->Fill(TrkTruthPhi);
                TrkTruthDxy_alljets[4]->Fill(TrkTruthDxy);
                TrkTruthDz_alljets[4]->Fill(TrkTruthDz);
                TrkTruthHitAll_alljets[4]->Fill(TrkTruthnHitAll);
                TrkTruthHitPixel_alljets[4]->Fill(TrkTruthnHitPixel);
                TrkTruthHitStrip_alljets[4]->Fill(TrkTruthnHitStrip);
            }
        }
        //Fake
        else if ( !theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::Fake] ) {
            nseltracksCat[5] += 1;
            TrkPt_alljets[5]->Fill(TrkPt);
            TrkEta_alljets[5]->Fill(TrkEta);
            TrkPhi_alljets[5]->Fill(TrkPhi);
            TrkDxy_alljets[5]->Fill(TrkDxy);
            TrkDz_alljets[5]->Fill(TrkDz);
            TrkHitAll_alljets[5]->Fill(TrknHitAll);
            TrkHitPixel_alljets[5]->Fill(TrknHitPixel);
            TrkHitStrip_alljets[5]->Fill(TrknHitStrip);
            // NO TRUTH FOR FAKES!!!
        }
    
    

    }
    // -------- END Loop Over (selected) Tracks ----------

    // Still have to fill some jet-flavour specific variables
    if (flav == 5){
        nTrkAll_bjet->Fill(nseltracks);
        //nTrkTruthAll_bjet->Fill(nseltracksTruth);
        for (unsigned int i = 0; i < TrkHistCat.size(); i++){
            nTrk_bjet[i]->Fill(nseltracksCat[i]);
            //nTrkTruth_bjet[i]->Fill(nseltracksTruthCat[i]);
        }
    }
    else if (flav == 4){
        nTrkAll_cjet->Fill(nseltracks);
        //nTrkTruthAll_cjet->Fill(nseltracksTruth);
        for (unsigned int i = 0; i < TrkHistCat.size(); i++){
            nTrk_cjet[i]->Fill(nseltracksCat[i]);
            //nTrkTruth_cjet[i]->Fill(nseltracksTruthCat[i]);
        }
    }
    else {
        nTrkAll_dusgjet->Fill(nseltracks);
        //nTrkTruthAll_dusgjet->Fill(nseltracksTruth);
        for (unsigned int i = 0; i < TrkHistCat.size(); i++){
            nTrk_dusgjet[i]->Fill(nseltracksCat[i]);
            //nTrkTruth_dusgjet[i]->Fill(nseltracksTruthCat[i]);
        }
    }

  }
  // -------- END Loop Over Jets ----------


}





//define this as a plug-in
DEFINE_FWK_MODULE(BDHadronTrackMonitoringAnalyzer);
