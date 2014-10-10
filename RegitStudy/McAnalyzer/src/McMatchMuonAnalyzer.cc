/*  Implementation: 
Analyzer to loop over different muon collections and do the MC hit-by-hit matching and analysis.
*/


// framework includes
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

// data formats
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// pixel
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"

// strip
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"
#include "DataFormats/SiStripDetId/interface/TIBDetId.h"
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"

//geometry

#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelTopology.h"

// sim data formats
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"

#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"

//tracking tools
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "RecoVertex/KalmanVertexFit/interface/SingleTrackVertexConstraint.h"

#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"

// ROOT
#include "TROOT.h"
#include "TNtuple.h"
#include "TLorentzVector.h"

// miscellaneous  
#include <fstream>
using namespace std;
using namespace reco;
const double m_mu = .105658;
const double epsilon = 0.001;
//__________________________________________________________________________
class McMatchMuonAnalyzer : public edm::EDAnalyzer
{

 public:
  explicit McMatchMuonAnalyzer(const edm::ParameterSet& pset);
  ~McMatchMuonAnalyzer();
  
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void beginJob(const edm::EventSetup& es);
  virtual void endJob();
  
private:
  void dummVectorEntry(vector<float>& result, Int_t entries);
  
  void fillRecoDimuonTuple(const edm::RefToBase<Track>& trkRef,reco::RecoToSimCollection& reco2Sim,vector<float>& result);
  void fillRecoTrackTuple(const edm::RefToBase<Track>& trkRef,vector<float>& result);
  void fillMatchedTrackTuple(const edm::RefToBase<Track>& trkRef,Int_t& nmatches,vector<float>& result);
  void fillTrackTuple(const TrackingParticleRef& trkRef, vector<float>& result);

  edm::RefToBase<Track> findRecoTrackMatch(const TrackingParticleRef& trk,reco::SimToRecoCollection& reco2Sim,Int_t& nmatches, Float_t& fFracShared);
  TrackingParticleRef   findSimTrackMatch(const edm::RefToBase<Track>& trkRef,reco::RecoToSimCollection& reco2Sim,Int_t& nmatches,Float_t& fFracShared);
    
  Int_t  getDetLayerId(const PSimHit& simHit);
  Int_t  getNumberOfPixelHits(const TrackingParticleRef& simTrack);
  Int_t  getNumberOfSimHits(const TrackingParticle& simTrack);
  Int_t  getSimParentId(const TrackingParticleRef& trk);
  Int_t  layerFromDetid(const DetId& detId); 
 
  void matchRecoMuons(edm::Handle<reco::MuonTrackLinksCollection>& muCollection,reco::RecoToSimCollection& reco2sim);
  void matchSimMuons(edm::Handle<TrackingParticleCollection>& simCollection,reco::SimToRecoCollection& sim2reco);

  void matchRecoDimuons(edm::Handle<reco::MuonTrackLinksCollection>& muCollection,reco::RecoToSimCollection& reco2sim);
  void matchSimDimuons(edm::Handle<TrackingParticleCollection>& simCollection,edm::Handle<reco::MuonTrackLinksCollection>& muCollection,
		       edm::Handle<edm::HepMCProduct>& hepEvt, reco::SimToRecoCollection& sim2reco); 

  void matchRecoTracks(edm::Handle<edm::View<Track> >& trackCollection, reco::RecoToSimCollection& p);
  void matchSimTracks(edm::Handle<TrackingParticleCollection>& simCollection, reco::SimToRecoCollection& q); 
  
  float refitWithVertex(const Track& recTrack);

  std::vector<PSimHit> getPSimHits(const TrackingParticleRef& st, DetId::Detector);
  std::vector<PSimHit> getPSimHits(const TrackingParticleRef& st);
  
  // ----- member data -----
  bool                         doreco2sim;
  bool                         dosim2reco;
  bool                         matchdimuons;
  bool                         matchmuons;
  bool                         matchstamuons;
  bool                         matchtrackertracks;
  double                       matchhitfraction;
  double                       minpttrackertrkcut;
  Int_t                        proc,ntrk,nvtx;
  int                          pdgdimuon;
  TNtuple                     *pnRecoSimDimuons;
  TNtuple                     *pnRecoSimMuons;
  TNtuple                     *pnRecoSimTracks;
  TNtuple                     *pnSimRecoDimuons;
  TNtuple                     *pnSimRecoMuons;
  TNtuple                     *pnSimRecoTracks;
  TNtuple                     *pnEventInfo;

  const TrackAssociatorByHits *theAssociatorByHits;
  const TrackerGeometry       *theTracker;
  const TransientTrackBuilder *theTTBuilder;
  const reco::BeamSpot        *theBeamSpot;
  const reco::VertexCollection *vertices;

  const SimHitTPAssociationProducer::SimHitTPAssociationList *simHitsTPAssoc;

  edm::InputTag                muontag;
  edm::InputTag                muonmaptag;
  edm::InputTag                simtracktag;
  edm::InputTag                statracktag; 
  edm::InputTag                trktracktag; 
  edm::InputTag                simHitTpMapTag; 
  edm::ParameterSet            paramset;
 
};


//___________________________________________________________________________
McMatchMuonAnalyzer::McMatchMuonAnalyzer(const edm::ParameterSet& pset):
  doreco2sim(pset.getParameter<bool>("doReco2Sim") ),
  dosim2reco(pset.getParameter<bool>("doSim2Reco") ),
  matchdimuons(pset.getParameter<bool>("matchDimuons") ),
  matchmuons(pset.getParameter<bool>("matchMuons") ),
  matchstamuons(pset.getParameter<bool>("matchStaMuons") ),
  matchtrackertracks(pset.getParameter<bool>("matchTrackerTracks") ),
  matchhitfraction(pset.getParameter<double>("matchHitFraction") ),
  minpttrackertrkcut(pset.getParameter<double>("minPtTrackerTrkCut") ),
  pdgdimuon(pset.getParameter<int>("pdgDimuon") ),
  muontag(pset.getUntrackedParameter<edm::InputTag>("muonTag") ),
  muonmaptag(pset.getUntrackedParameter<edm::InputTag>("muonMapTag") ),
  simtracktag(pset.getUntrackedParameter<edm::InputTag>("simtracks") ),
  trktracktag(pset.getUntrackedParameter<edm::InputTag>("trktracks") ),
  simHitTpMapTag(pset.getUntrackedParameter<edm::InputTag>("tagForHitTpMatching") ),
  paramset(pset)
{
  // constructor
 
 
}


//_____________________________________________________________________________
McMatchMuonAnalyzer::~McMatchMuonAnalyzer()
{
  // destructor

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//_____________________________________________________________________
void McMatchMuonAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  // method called each event  
 
  edm::LogInfo("MCMatchAnalyzer")<<"Start analyzing each event ...";

  // Get generated event
  edm::Handle<edm::HepMCProduct> hepEv;
  ev.getByLabel("source",hepEv);
  proc = hepEv->GetEvent()->signal_process_id();
  edm::LogInfo("McMatchMuonAnalyzer::analyze()")<<"Process ID= " << hepEv->GetEvent()->signal_process_id();

  // Get simulated particles
  edm::Handle<TrackingParticleCollection> simCollection;
  ev.getByLabel(simtracktag,simCollection);
  edm::LogInfo("McMatchMuonAnalyzer::analyze()")<< "Size of simCollection = " << simCollection.product()->size();  

  // Get reconstructed tracks
  edm::Handle<edm::View<Track> >  trackCollection;
  ev.getByLabel(trktracktag, trackCollection); 
  edm::LogInfo("McMatchMuonAnalyzer::analyze()")<< "Size of trackCollection = " << trackCollection.product()->size();

 // Get muon tracks only
  edm::Handle<reco::MuonTrackLinksCollection> muTrkLksCollection;
  ev.getByLabel(muontag,muTrkLksCollection);
  const reco::MuonTrackLinksCollection mutracks = *(muTrkLksCollection.product());
  edm::LogInfo("McMatchAnalyzer::analyze()") << "##### Size of MuonTrackLinksCollection = "<< mutracks.size()<<endl;
  
  //get beamSpot
  edm::Handle<reco::BeamSpot>      beamSpotHandle;
  ev.getByLabel("offlineBeamSpot", beamSpotHandle);
  theBeamSpot = beamSpotHandle.product();

   // Get vertices
  edm::Handle<reco::VertexCollection> vertexCollection;
  ev.getByLabel("pixelVertices",vertexCollection);
  vertices = vertexCollection.product();
  edm::LogInfo("McMatchMuonAnalyzer::analyze()")<<"Number of vertices in the event = "<< vertexCollection.product()->size();

  //get sim hits - TP map
  edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssocHandle;
  ev.getByLabel(simHitTpMapTag,simHitsTPAssocHandle);
  simHitsTPAssoc = simHitsTPAssocHandle.product();

  // fill the event info TNtuple
  ntrk = trackCollection.product()->size();
  nvtx = vertexCollection.product()->size();
  {
    vector<float> result;
    result.push_back(proc); // proc
    result.push_back(ntrk); // ntrkr
    result.push_back(nvtx); // nvtxr
    
    pnEventInfo->Fill(&result[0]);
  }  

  if(dosim2reco)
    {
      if(matchtrackertracks)
	{
	  edm::LogInfo("McMatchMuonAnalyzer::analyze()")<<"Got track sim2reco association maps for tracks! ";
	  reco::SimToRecoCollection trkSim2Reco = theAssociatorByHits->associateSimToReco(trackCollection,simCollection,&ev,&es);
	  
	  matchSimTracks(simCollection, trkSim2Reco);
	}
      if(matchmuons || matchdimuons)
	{
	  edm::Handle<reco::SimToRecoCollection> muSimRecoCollection;
	  ev.getByLabel(muonmaptag,muSimRecoCollection);
	  if( muSimRecoCollection.isValid() )
	    {
	      reco::SimToRecoCollection muSim2Reco = *(muSimRecoCollection.product());
	      LogDebug("McMatchAnalyzer::analyze()") << "##### Size of muSim2Reco collection = "<<muSim2Reco.size();
	      
	      if(matchmuons) matchSimMuons(simCollection,muSim2Reco);
	      if(matchdimuons) matchSimDimuons(simCollection,muTrkLksCollection,hepEv,muSim2Reco);
	    }
	  else 
	    LogDebug("McMatchAnalyzer::analyze()") << "##### NO SimToRecoCollection found for muons!";
	}
    }
  if(doreco2sim)
    {
      if(matchtrackertracks)
	{
	  edm::LogInfo("McMatchMuonAnalyzer::analyze()")<<"Got track reco2sim association maps for tracks! ";
	  reco::RecoToSimCollection trkReco2Sim = theAssociatorByHits->associateRecoToSim(trackCollection,simCollection,&ev,&es);
      
	  matchRecoTracks(trackCollection,trkReco2Sim);
	}      
      if(matchmuons || matchdimuons)
	{
	  edm::Handle<reco::RecoToSimCollection> muRecoSimCollection;
	  ev.getByLabel(muonmaptag,muRecoSimCollection);
	  if(muRecoSimCollection.isValid() )
	    {
	      reco::RecoToSimCollection muReco2Sim = *(muRecoSimCollection.product());
	      edm::LogInfo("McMatchAnalyzer::analyze()") <<"##### Size of muReco2Sim collection = "<<muReco2Sim.size()<<endl;
	      
	      if(matchmuons) matchRecoMuons(muTrkLksCollection,muReco2Sim);
	      if(matchdimuons && muTrkLksCollection.product()->size() > 1 ) matchRecoDimuons(muTrkLksCollection,muReco2Sim);
	    }
	  else 
	    LogDebug("McMatchAnalyzer::analyze()") << "##### NO RecoToSimCollection found for muons!";
	}
    }

  edm::LogInfo("McMatchMuonAnalyzer::analyze()") << "[McMatchMuonAnalyzer] done with event# " << ev.id();
}

//____________________________________________________________________________
void McMatchMuonAnalyzer::beginJob(const edm::EventSetup& es)
{
 // method called once each job just before starting event loop
  edm::LogInfo("MCMatchAnalyzer::beginJob()")<<"Begin job initialization ...";

  // Get tracker geometry
  edm::ESHandle<TrackerGeometry> tracker;
  es.get<TrackerDigiGeometryRecord>().get(tracker);
  theTracker = tracker.product();
 
  // Get track associator by hits
  edm::ESHandle<TrackAssociatorBase> theHitsAssociator;
  es.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",theHitsAssociator);
  theAssociatorByHits     = (const TrackAssociatorByHits*)theHitsAssociator.product();

  // Get transient track builder
  edm::ESHandle<TransientTrackBuilder> builder;
  es.get<TransientTrackRecord>().get("TransientTrackBuilder", builder);
  theTTBuilder = builder.product();
 
  // Bookkeepping
  // first global muon information, then sta muon, tracker muon info, and last simu info
  // the muon matching: tracker + sta hit-by-hit
  edm::Service<TFileService>   fs;
  pnRecoSimMuons   = fs->make<TNtuple>("pnRecoSimMuons","pnRecoSimMuons",
				       "charge:chi2:chi2ndof:dxy:dxyerr:dz:dzerr:nvalidhits:eta:phi:pt:" // reco tracker muon  11
				       "nmatch:frachitmatch:" // #of reco trks amtched to one sim track   2
				       "idsim:idparentsim:chargesim:etasim:phisim:ptsim");// matched sim track   6
  pnRecoSimDimuons = fs->make<TNtuple>("pnRecoSimDimuons","pnRecoSimDimuons",
				       "y:minv:phi:pt:" // dimuon 4
				       "eta1:phi1:pt1:nmatch1:frachitmatch1:idsim1:idparentsim1:"//mu1 7
				       "eta2:phi2:pt2:nmatch2:frachitmatch2:idsim2:idparentsim2");//mu2 7
			  

  pnSimRecoDimuons = fs->make<TNtuple>("pnSimRecoDimuons","pnSimRecoDimuons",
				       "y:minv:phi:pt:" // sim Z0  4
				       "charge1:eta1:phi1:pt1:npixelhits1:" // sim muon 1  5
				       "nmatch1:frachitmatch1:chargereco1:etareco1:phireco1:ptreco1:" // reco tracker  muon1   6
				       "charge2:eta2:phi2:pt2:npixelhits2:" // sim muon 2   5
				       "nmatch2:frachitmatch2:chargereco2:etareco2:phireco2:ptreco2:" // reco tracker muon2   6
				       "yreco:minvreco:phireco:ptreco"); // reconstructed Z0  4
				     
  pnSimRecoMuons  = fs->make<TNtuple>("pnSimRecoMuons","pnSimRecoMuons",
				      "id:idparent:charge:eta:phi:pt:npixelhits:"  //sim track   7
				      "nmatch:frachitmatch:" //sim-reco: 2
				      "chargereco:etareco:phireco:ptreco"); // reco tracker muon  4
				    
  pnRecoSimTracks  = fs->make<TNtuple>("pnRecoSimTracks","pnRecoSimTracks",
				       "charge:chi2:chi2ndof:dxy:dxyerr:dz:dzerr:nvalidhits:eta:phi:pt:" //reco track
				       "nmatch:frachitmatch:"// #of reco trks matched to one sim track
				       "idsim:idparentsim:chargesim:etasim:phisim:ptsim"); // matched sim track

  pnSimRecoTracks  = fs->make<TNtuple>("pnSimRecoTracks","pnSimRecoTracks",
				       "id:idparent:charge:eta:phi:pt:npixelhits:status:"  //sim track
				       "nmatch:frachitmatch:" //# of reco tracks matched to one sim track
				       "chargereco:etareco:phireco:ptreco"); // matched reco track
  pnEventInfo      = fs->make<TNtuple>("pnEventInfo","pnEventInfo","proc:ntrk:nvtx");
  
  edm::LogInfo("MCMatchAnalyzer::beginJob()")<<"Ended job initialization ...";

}


//_________________________________________________________________________
void McMatchMuonAnalyzer::endJob()
{
  //method called once each job just after ending the event loop 

}


//________________________________________________________________________
void McMatchMuonAnalyzer::matchRecoMuons(edm::Handle<reco::MuonTrackLinksCollection>& muTrkLksCollection, 
					 reco::RecoToSimCollection& muReco2Sim)
{
  // fill the TNtuple pnRecoSimMuons, -- 19 fields
  // with reco and corresponding sim tracks information

//  pnRecoSimMuons   = fs->make<TNtuple>("pnRecoSimMuons","pnRecoSimMuons",
// 				       "charge:chi2:chi2ndof:dxy:dxyerr:dz:dzerr:nvalidhits:eta:phi:pt:" // reco tracker muon  11
// 				       "nmatch:frachitmatch:" // #of reco trks amtched to one sim track   2
// 				       "idsim:idparentsim:chargesim:etasim:phisim:ptsim");// matched sim track   6


  edm::LogInfo("MCMatchAnalyzer::matchRecoMuons()")<<"Matching recoMuons ...";
  for ( reco::MuonTrackLinksCollection::const_iterator ilink = muTrkLksCollection->begin(); 
	ilink != muTrkLksCollection->end(); ilink++) 
    {    
      // chose the muon
      edm::RefToBase<reco::Track> trkMuRef;
      if( matchstamuons ) trkMuRef = edm::RefToBase<reco::Track>(ilink->standAloneTrack());
      else trkMuRef = edm::RefToBase<reco::Track>(ilink->globalTrack());
      if( trkMuRef.isNull() ) continue;
    
      vector<float> result1;
      LogDebug("McMatchMuonAnalyzer::matchRecoMuons()")<<"##### Global track: pT= "<<trkMuRef->pt()<<endl;
      fillRecoTrackTuple(trkMuRef,result1);
 
      // ... matched sim
      Int_t nSimMu       = 0;
      Float_t fHitFracMu = 0.;
      TrackingParticleRef matchedSimMuTrack = findSimTrackMatch(trkMuRef,muReco2Sim,nSimMu,fHitFracMu);
      
      result1.push_back(nSimMu); // # of sim tracks matched to one reco mu_tracker track
      result1.push_back(fHitFracMu);
      
      if( !matchedSimMuTrack.isNull() && nSimMu!=0)
	{
	  Int_t ids    = matchedSimMuTrack->pdgId();
	  Int_t parent = getSimParentId(matchedSimMuTrack);
	  result1.push_back(ids);
	  result1.push_back(parent);
	  fillTrackTuple(matchedSimMuTrack,result1); // ch, eta, phi, pt
	}
      else dummVectorEntry(result1,6);
      
      
      // fill
      pnRecoSimMuons->Fill(&result1[0]);
      result1.clear();
    }// muon loop

  edm::LogInfo("MCMatchAnalyzer::matchRecoMuons()")<<"Done matchRecoMuons()!";
	
}


//_________________________________________________________________________
void McMatchMuonAnalyzer::matchSimMuons(edm::Handle<TrackingParticleCollection>& simCollection,
					reco::SimToRecoCollection& sim2recoCollection)
{
  // fill TNtuple pnSimRecoMuons  --- 11 fields
// pnSimRecoMuons  = fs->make<TNtuple>("pnSimRecoMuons","pnSimRecoMuons",
// 				      "id:idparent:charge:eta:phi:pt:npixelhits:"  //sim track   7
// 				      "chargereco:etareco:phireco:ptreco"); // reco tracker muon  4

  edm::LogInfo("MCMatchAnalyzer::matchSimMuons()")<<"Start matching sim muons ...";
     
  for(TrackingParticleCollection::size_type i=0; i < simCollection.product()->size(); i++)
    {
      const TrackingParticleRef simTrack(simCollection, i);
      if( simTrack.isNull() ) continue; 
      const SimTrack *gTrack = &(*simTrack->g4Track_begin());
      if( gTrack == NULL ) continue;

      // only muons
      if( abs(gTrack->type()) != 13 ) continue; 

      edm::LogInfo("MCMatchAnalyzer::matchSimMuons()")<<"Got an fmuon!.";
      Int_t nPixelLayers = getNumberOfPixelHits(simTrack);
  
      // there is no tracker track if not a seed of 3 pixel hits; hence a condition of matching sta+trk will fail
      // make the cut in the plotter and only for the HI case      
      // if( nPixelLayers < 3 ) continue;
      
      vector<float> result30;
      result30.push_back(simTrack->pdgId());    
       
      Int_t parent = getSimParentId(simTrack);
      result30.push_back(parent) ;

      fillTrackTuple(simTrack,result30);

      result30.push_back(nPixelLayers); 
     
      Int_t nRecoMu = 0;   
      Float_t fFracReco = 0.;   
      edm::RefToBase<Track>  matchedRecMuTrk = findRecoTrackMatch(simTrack,sim2recoCollection,nRecoMu,fFracReco);
      result30.push_back(nRecoMu); // # of reco tracks matched to one sim track
      result30.push_back(fFracReco); // fraction of matched hits
      if( !matchedRecMuTrk.isNull() && nRecoMu!=0 )
	  fillMatchedTrackTuple(matchedRecMuTrk,nRecoMu,result30);
      else 
	dummVectorEntry(result30,4);
      
      pnSimRecoMuons->Fill(&result30[0]);
      result30.clear();
      
    }// TrackParticleCollection loop
  
  edm::LogInfo("MCMatchAnalyzer::matchSimDimuons()")<<"Done matchSiMuons()!.";
}


//______________________________________________
void McMatchMuonAnalyzer::matchRecoDimuons(edm::Handle<reco::MuonTrackLinksCollection>& muTrkLksCollection, 
					   reco::RecoToSimCollection& trkReco2Sim)
{
  // fill the TNtuple pnRecoSimDimuons,   18 fields
  // with reco and corresponding sim tracks information
//  pnRecoSimDimuons = fs->make<TNtuple>("pnRecoSimDimuons","pnRecoSimDimuons",
// 				       "y:minv:phi:pt:" // dimuon 4
// 				       "eta1:phi1:pt1:nmatch1:frachitmatch1:idsim1:idparentsim1:"//mu1 7
// 				       "eta2:phi2:pt2:nmatch2:frachitmatch2:idsim2:idparentsim2");//mu2 7

  edm::LogInfo("McMatchMuonAnalyzer::matchRecoDimuons()")<<"Start reconstructing Dimuons!";
  
  for ( reco::MuonTrackLinksCollection::const_iterator links1 = muTrkLksCollection->begin(); 
	links1 != muTrkLksCollection->end()-1; links1++) 
    {
      edm::RefToBase<reco::Track> trkMuRef1;
      if( matchstamuons ) trkMuRef1 = edm::RefToBase<reco::Track>(links1->standAloneTrack());
      else trkMuRef1 = edm::RefToBase<reco::Track>(links1->globalTrack());
      if( trkMuRef1.isNull() ) continue;

      for ( reco::MuonTrackLinksCollection::const_iterator links2 = links1+1; 
	    links2 != muTrkLksCollection->end(); links2++) 
	{
	  edm::RefToBase<reco::Track> trkMuRef2;
	  if( matchstamuons ) trkMuRef2 = edm::RefToBase<reco::Track>(links2->standAloneTrack());
	  else trkMuRef2 = edm::RefToBase<reco::Track>(links2->globalTrack());
	  if( trkMuRef2.isNull() ) continue;

	  if(trkMuRef1->charge()*trkMuRef2->charge()>0) continue; // just oppposite sign dimuons
	  
	  vector<float> result;
	  
	  TLorentzVector child1, child2;
	  double en1 = sqrt(trkMuRef1->p()*trkMuRef1->p()+m_mu*m_mu);
	  double en2 = sqrt(trkMuRef2->p()*trkMuRef2->p()+m_mu*m_mu);

	  child1.SetPxPyPzE(trkMuRef1->px(),trkMuRef1->py(),trkMuRef1->pz(),en1);
	  child2.SetPxPyPzE(trkMuRef2->px(),trkMuRef2->py(),trkMuRef2->pz(),en2);
	  
	  TLorentzVector dimuon;
	  dimuon = child1 + child2;
	 	  
	  result.push_back(dimuon.Rapidity());
	  result.push_back(dimuon.M());
	  result.push_back(dimuon.Phi());
	  result.push_back(dimuon.Pt());
	  
	  // ######### first muon  
	  result.push_back(trkMuRef1->eta());
	  result.push_back(trkMuRef1->phi());
	  result.push_back(trkMuRef1->pt());
	  fillRecoDimuonTuple(trkMuRef1,trkReco2Sim,result);
	 
	  // second muon
	  result.push_back(trkMuRef2->eta());
	  result.push_back(trkMuRef2->phi());
	  result.push_back(trkMuRef2->pt());
	  fillRecoDimuonTuple(trkMuRef2,trkReco2Sim,result);
	  
	  LogDebug("McMatchMuonAnalyzer::matchRecoDimuons()")<<"First sta,trk muon filled!";
	  pnRecoSimDimuons->Fill(&result[0]);

	  LogDebug("McMatchMuonAnalyzer::matchRecoDimuons():result")<<"eta mu1 = "<<result[4]<<"\t eta mu2 = "<< result[13];
	  LogDebug("McMatchMuonAnalyzer::matchRecoDimuons():direct")<<"eta mu1 = " << trkMuRef1->eta() <<"\t ta mu2 = " << trkMuRef2->eta();
	  result.clear();
       }// second muon loop
    }// first muon loop
  edm::LogInfo("McMatchMuonAnalyzer::matchRecoDimuons()")<<"Done matchRecoDimuons()!";
}


//_________________________________________________________________________
void McMatchMuonAnalyzer::matchSimDimuons(edm::Handle<TrackingParticleCollection>& simCollection,
					  edm::Handle<reco::MuonTrackLinksCollection>& muTrkLksCollection,
					  edm::Handle<edm::HepMCProduct>& hepEvt, reco::SimToRecoCollection& sim2RecoTrk)
{
  // fill TNtuple pnSimRecoDimuons with gen info for the gen Z0 and muon daughters and reco info for the reco muons

  // status == 1 : stable particle at pythia level, but which can be decayed by geant
  // status == 2 :
  // status == 3 : outgoing partons from hard scattering

  // The barcode is the particle's reference number, every vertex in the
  // event has a unique barcode. Particle barcodes are positive numbers,
  // vertex barcodes are negative numbers.

//  pnSimRecoDimuons = fs->make<TNtuple>("pnSimRecoDimuons","pnSimRecoDimuons",
// 				       "y:minv:phi:pt:" // sim Z0  4
// 				       "charge1:eta1:phi1:pt1:npixelhits1:" // sim muon 1  5
// 				       "nmatch1:frachitmatch1:chargereco1:etareco1:phireco1:ptreco1:" // reco tracker  muon1   6
// 				       "charge2:eta2:phi2:pt2:npixelhits2:" // sim muon 2   5
// 				       "nmatch2:frachitmatch2:chargereco2:etareco2:phireco2:ptreco2:" // reco tracker muon2   6
// 				       "yreco:minvreco:phireco:ptreco"); // reconstructed Z0  4

  edm::LogInfo("MCMatchAnalyzer::matchSimDimuons()")<<"Start matching dimuons ...";
  
  const HepMC::GenEvent * myGenEvent = hepEvt->GetEvent();

  Bool_t foundmuons = false;
  vector<Int_t> ixmu;
  vector<float> result3;
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) 
    { 
      if ( ( abs((*p)->pdg_id()) == pdgdimuon ) && (*p)->status() == 3 ) 
	{ 
	  for( HepMC::GenVertex::particle_iterator aDaughter=(*p)->end_vertex()->particles_begin(HepMC::descendants); 
	       aDaughter !=(*p)->end_vertex()->particles_end(HepMC::descendants);
	       aDaughter++)
	    {
	      if ( abs((*aDaughter)->pdg_id())==13 && (*aDaughter)->status()==1 )
		    
		{
		  ixmu.push_back((*aDaughter)->barcode());
		  LogDebug("MCMatchAnalyzer::matchSimDimuons()") << "Stable muon from Z0" << "\tindex= "<< (*aDaughter)->barcode(); 
		  foundmuons = true;
		}//  z0 descendentes
	    }
	  if (foundmuons)
	    {
	      double rapidity = 0.5 * log(( (*p)->momentum().e()+(*p)->momentum().pz() )/( (*p)->momentum().e()-(*p)->momentum().pz() ));
	      result3.push_back(rapidity);
	      result3.push_back((*p)->momentum().m());
	      result3.push_back((*p)->momentum().phi());
	      result3.push_back((*p)->momentum().perp());
	    }
	}// Z0
    }// loop over genParticles

  if( foundmuons )
    {
    
      vector<edm::RefToBase<Track> > trkstub;
      Int_t ndaughters  = 0;

      for(TrackingParticleCollection::size_type i=0; i < simCollection.product()->size(); i++)
	{
	  const TrackingParticleRef simTrk(simCollection, i);
	  if( simTrk.isNull() ) continue;
	  const SimTrack *gTrack = &(*simTrk->g4Track_begin());
	  if( gTrack == NULL ) continue;
	
	  Int_t nPixelLayers = getNumberOfPixelHits(simTrk);
	  if(abs(gTrack->type())==13 && (gTrack->genpartIndex()==ixmu[0] || gTrack->genpartIndex()==ixmu[1] ))
	    {
	      edm::LogInfo("MCMatchAnalyzer::matchSimDimuons()")<<"Got an xmuon!.";

	      // sim track
	      fillTrackTuple(simTrk,result3);
	      result3.push_back(nPixelLayers);

	      // staMatched_muon
	      Int_t nMu = 0;    
	      Float_t fHitMu = 0.;  
	      edm::RefToBase<Track>  matchedRec = findRecoTrackMatch(simTrk,sim2RecoTrk,nMu,fHitMu);
	      result3.push_back(nMu); // # of reco tracks matched to one sim track
	      result3.push_back(fHitMu);
	      if( !matchedRec.isNull() && nMu!=0 )
		fillMatchedTrackTuple( matchedRec,nMu,result3);
	      else 
		dummVectorEntry(result3,4);
		
	      if( (!matchedRec.isNull() && nMu!=0)  )
		{
		  trkstub.push_back(matchedRec);
		  ndaughters++;
		}
	      
	    }// muon id
	}// TrackCollection loop

      if(ndaughters == 1) // not found second daughter
	{
	  LogDebug("MCMatchAnalyzer::matchSimDimuons()")<<"Only one sim daughter found to match a reco track"<<endl;
	  dummVectorEntry(result3,15); 

	}
      if(ndaughters == 0) // no daughter found ?? check this
	{
	  LogDebug("MCMatchAnalyzer::matchSimDimuons()")<<"No sim daughter reconstructed"<<endl;
	  dummVectorEntry(result3,26); //(2*5 for muons + 2*6reco muons+4 reco dimuon)
	}

      // get the reco Z0
      if(ndaughters == 2)
	{
	  edm::LogInfo("MCMatchAnalyzer::matchSimDimuons()")<<"Both sim daughters found to match a reco track"<<endl;

	  vector<edm::RefToBase<Track> > glbchild;
	  for( Int_t istubs = 0; istubs < ndaughters; istubs++)
	    {
	      if( trkstub[istubs].isNull() ) continue;
	      for ( reco::MuonTrackLinksCollection::const_iterator links = muTrkLksCollection->begin(); 
		    links != muTrkLksCollection->end(); links++) 
		{
		  edm::RefToBase<Track> trkMuRef;
		  if( matchstamuons ) trkMuRef = edm::RefToBase<reco::Track>(links->standAloneTrack());
		  else trkMuRef = edm::RefToBase<reco::Track>(links->globalTrack());
		  if( trkMuRef.isNull() ) continue;
		  
		  bool match = (abs(trkMuRef->phi() == trkstub[istubs]->phi()) ) &&
		    (abs(trkMuRef->eta() == trkstub[istubs]->eta()) ) ;
		  if(!match) continue;
		  
		  glbchild.push_back(trkMuRef);
		  break; // once found a match, stop it, and go to th next stub
		}// each muon
	    }//each daughter

	  if(glbchild.size()==2)
	    {
	      edm::LogInfo("MCMatchAnalyzer::matchSimDimuons()")<<"Fill reco_dimuon and reco_kids info";
	      edm::RefToBase<Track> ch1 = glbchild[0];
	      edm::RefToBase<Track> ch2 = glbchild[1];
	      edm::LogInfo("MCMatchAnalyzer::matchSimDimuons()")<<"Got the kids from the vector of kids";
	    	    
	      // reconstructed Z0
	      TLorentzVector child1, child2;
	      double en1 = sqrt(ch1->p()*ch1->p()+m_mu*m_mu); 
	      double en2 = sqrt(ch2->p()*ch2->p()+m_mu*m_mu);
	      
	      child1.SetPxPyPzE(ch1->px(),ch1->py(),ch1->pz(),en1);
	      child2.SetPxPyPzE(ch2->px(),ch2->py(),ch2->pz(),en2);
	      
	      
	      TLorentzVector recoz0;
	      recoz0 = child1 + child2;
	      result3.push_back(recoz0.Rapidity());
	      result3.push_back(recoz0.M());
	      result3.push_back(recoz0.Phi());
	      result3.push_back(recoz0.Pt());
	      
	      edm::LogInfo("MCMatchAnalyzer::matchSimDimuons()")<<"Filled the matched reco Z0 info";
	   
	    }
	  // just in case
	  if(glbchild.size()==1) dummVectorEntry(result3,4); 
	

	  glbchild.clear();
	}//ndaughters>=1
    
      trkstub.clear();
      pnSimRecoDimuons->Fill(&result3[0]);
      result3.clear();
    }// there are Z0 sim muons; foundmuons
  edm::LogInfo("MCMatchAnalyzer::matchSimDimuons()")<<"Done matchSimDimuons()!.";
}


//________________________________________________________________________
void McMatchMuonAnalyzer::matchRecoTracks(edm::Handle<edm::View<Track> >& trackCollection,
					  reco::RecoToSimCollection& p)
{
  // fill the TNtuple pnRecoSimTracks, 
  // with reco tracks and their corresponding sim tracks information

// pnRecoSimTracks  = fs->make<TNtuple>("pnRecoSimTracks","pnRecoSimTracks",
// 				       "charge:chi2:chi2ndof:dxy:dxyerr:dz:dzerr:nvalidhits:eta:phi:pt:" //reco track
// 				       "nmatch:frachitmatch:"// #of reco trks matched to one sim track
// 				       "idsim:idparentsim:chargesim:etasim:phisim:ptsim"); // matched sim track
  
  edm::LogInfo("MCMatchAnalyzer::matchRecoTracks()")<<"Start matching reco tracks ...";

  for(edm::View<Track>::size_type i=0; i < trackCollection.product()->size(); ++i)
  {
    edm::RefToBase<Track> recTrack(trackCollection, i);
    if ( recTrack.isNull() || recTrack->pt() < minpttrackertrkcut ) continue;
    vector<float> result2;
    Int_t ids      = -99;
    Int_t parentId = -99;
    Int_t nSim  = 0;
    Float_t fFr = 0.;
    // reco tracker
    fillRecoTrackTuple(recTrack,result2);
    TrackingParticleRef matchedSimTrack = findSimTrackMatch(recTrack,p,nSim,fFr);
    result2.push_back(nSim); // # matched sim track to one reco track
    result2.push_back(fFr);

    if( !matchedSimTrack.isNull() && nSim!=0)
      {
	// result2.push_back(refitWithVertex(*recTrack));                 // ptv
	// matched sim 
	
	ids      = matchedSimTrack->pdgId();
	parentId = getSimParentId(matchedSimTrack);
	result2.push_back(ids);
	result2.push_back(parentId);
	fillTrackTuple(matchedSimTrack,result2);
      }// there is a matched track
    else dummVectorEntry(result2,6);
    
    pnRecoSimTracks->Fill(&result2[0]);
    result2.clear();
  }//i trackCollection
  edm::LogInfo("MCMatchAnalyzer::matchRecoTracks()")<<"Done matchRecoTracks().";
}


//_________________________________________________________________________
void McMatchMuonAnalyzer::matchSimTracks(edm::Handle<TrackingParticleCollection>& simCollection,
					 reco::SimToRecoCollection& q)
{
  // fill TNtuple pnSimRecoTracks with sim track and corresponding reco tracks info
 
// pnSimRecoTracks  = fs->make<TNtuple>("pnSimRecoTracks","pnSimRecoTracks",
// 				       "id:idparent:charge:eta:phi:pt:npixelhits:status:"  //sim track
// 				       "nmatch:frachitmatch:" //# of reco tracks matched to one sim track
// 				       "chargereco:etareco:phireco:ptreco"); // matched reco track
  
  edm::LogInfo("MCMatchAnalyzer::matchSimTracks()")<<"Start matching simTracks ...";
 
  for(TrackingParticleCollection::size_type i=0; i < simCollection.product()->size(); i++)
    {
      const TrackingParticleRef simTrack(simCollection,i);
      if( simTrack.isNull() ) continue;
      if( simTrack->charge() == 0 || simTrack->pt() < minpttrackertrkcut ) continue;
      if(simTrack->pt() > 3e+4) continue;
      Int_t nPixelLayers = getNumberOfPixelHits(simTrack);
      if( nPixelLayers < 3 ) continue;
      
      vector<float> result4;
      
      // sim track
      result4.push_back(simTrack->pdgId());               // id
      Int_t parent = getSimParentId(simTrack);
      
      result4.push_back(parent) ;
      fillTrackTuple(simTrack,result4);
      result4.push_back(nPixelLayers); // nhits
      result4.push_back(simTrack->status() );
      // recoTracker matched 
      Int_t nRec = 0;
      Float_t fF = 0.;
      edm::RefToBase<Track> matchedRecTrack = findRecoTrackMatch(simTrack,q,nRec,fF);
      result4.push_back(nRec); // # of reco tracks matched to one sim track
      result4.push_back(fF);

      if( !matchedRecTrack.isNull() && nRec!=0)
	  fillMatchedTrackTuple(matchedRecTrack,nRec,result4);
      else 
	  dummVectorEntry(result4,4);
      
      // fill
      pnSimRecoTracks->Fill(&result4[0]);
      result4.clear();
    }

  edm::LogInfo("MCMatchAnalyzer::matchSimTracks()")<<"Done matchSimTracks()!";
}


//_______________________________________________________________________
void McMatchMuonAnalyzer::fillRecoDimuonTuple(const edm::RefToBase<Track>& trkRef,
					  reco::RecoToSimCollection& reco2Sim,
					  vector<float>& result)
{
  // get the match, 
  // fill the result with the id of match, id of parent, and and number of matches
  edm::LogInfo("MCMatchAnalyzer::fillRecoDimuonTuple()")<<"Filling dimuon tuple ...";

  Int_t nTrkSim = 0;
  Float_t fFraction = 0.;
  TrackingParticleRef matchedSim = findSimTrackMatch(trkRef,reco2Sim,nTrkSim,fFraction); 
  Int_t ids = -99;
  Int_t parentID = -99;

  if( !matchedSim.isNull() )
    {
      ids = matchedSim->pdgId();
      parentID = getSimParentId(matchedSim);
     
    }
 
  result.push_back(nTrkSim);
  result.push_back(fFraction);
  result.push_back(ids);
  result.push_back(parentID);
  
  edm::LogInfo("MCMatchAnalyzer::fillRecoDimuonTuple()")<<"Done fillRecoDimuonTuple()!";
 }


//_______________________________________________________________________
void McMatchMuonAnalyzer::fillRecoTrackTuple(const edm::RefToBase<Track>& trkRef, vector<float>& result)
{
  // fill reco track info to be fed later to the tuple
  edm::LogInfo("MCMatchAnalyzer::fillRecoTrackTuple()")<<"Filling reco track info ...";
  // transverse and z DCA calculated with respect to the beamSpot plus the error

  if( !trkRef.isNull())
    {
      // it is not DCA 
      double dt      = trkRef->dxy();
      double sigmaDt = trkRef->dxyError();
      double dz      = trkRef->dz();
      double sigmaDz = trkRef->dzError();

     //  double dt      = trkRef->dxy(theBeamSpot->position());
//       double sigmaDt = sqrt(trkRef->dxyError() * trkRef->dxyError() +
// 			    theBeamSpot->BeamWidth() * theBeamSpot->BeamWidth());
      
      result.push_back(trkRef->charge());                         // charge
      result.push_back(trkRef->chi2());                           // normalized chi2
      result.push_back(trkRef->normalizedChi2());                 // normalized Chi2
      result.push_back(dt);                                       // transverse DCA
      result.push_back(sigmaDt);                                  // Sigma_Dca_t
      result.push_back(dz);                                       // z DCA
      result.push_back(sigmaDz);                                  // Sigma_Dca_z
      result.push_back(trkRef->numberOfValidHits());              // number of hits found 
      result.push_back(trkRef->eta());                            // eta
      result.push_back(trkRef->phi());                            // phi
      result.push_back(trkRef->pt());                             // pt
    }
  else dummVectorEntry(result,11);
  edm::LogInfo("MCMatchAnalyzer::fillRecoTrackTuple()")<<"Done fillRecoTrackTuple()!";
}


//____________________________________________________________________
void McMatchMuonAnalyzer::fillMatchedTrackTuple(const edm::RefToBase<Track>& recoTrack,
						Int_t& nMatches, vector<float>& result)
{
  // the info part of the matched sim track
  edm::LogInfo("MCMatchAnalyzer::fillMatchedTrackTuple()")<<"Filling matched reco track info ...";

  if(nMatches > 0 )
    {
      result.push_back(recoTrack->charge()); // chargereco
      result.push_back(recoTrack->eta());    // eta
      result.push_back(recoTrack->phi());    // phireco
      result.push_back(recoTrack->pt());     // ptreco
    }
  else dummVectorEntry(result,4);
  edm::LogInfo("MCMatchAnalyzer::fillMatchedTrackTuple()")<<"Done fillMatchedTrackTuple()!"; 
}


//____________________________________________________________________
void McMatchMuonAnalyzer::fillTrackTuple(const TrackingParticleRef& recoTrack,
				     vector<float>& result)
{
  // the info part of the matched sim track
  edm::LogInfo("MCMatchAnalyzer::fillMatchedTrackTuple()")<<"Filling matched reco track info ...";

  if( !recoTrack.isNull() )
    {
      result.push_back(recoTrack->charge()); // chargereco
      result.push_back(recoTrack->eta());    // eta
      result.push_back(recoTrack->phi());    // phireco
      result.push_back(recoTrack->pt());     // ptreco
    }
  else dummVectorEntry(result,4);
  edm::LogInfo("MCMatchAnalyzer::fillMatchedTrackTuple()")<<"Done fillMatchedTrackTuple()!"; 
}


//________________________________________________________________________
edm::RefToBase<Track> McMatchMuonAnalyzer::findRecoTrackMatch(const TrackingParticleRef& simTrack,
							      reco::SimToRecoCollection& sim2Reco, 
							      Int_t& nMatches, Float_t& fFracShared)
{
  // return the matched reco track
  edm::LogInfo("MCMatchAnalyzer::findRecoTrackMatch()")<<"Finding reco track match ...";

  edm::RefToBase<Track> recoTrackMatch;

  if(sim2Reco.find(simTrack) != sim2Reco.end()) 	 
    {
      vector<pair<edm::RefToBase<Track>, double> > recTracks = sim2Reco[simTrack];
      for(vector<pair<edm::RefToBase<Track>,double> >::const_iterator it = recTracks.begin(); 
	  it != recTracks.end(); it++)
	{
	  edm::RefToBase<Track> recTrack = it->first;
	  float fraction = it->second;
	  
	  if(fraction > matchhitfraction )
	    { 
	      recoTrackMatch = recTrack; 
	      nMatches++; 
	      fFracShared = fraction;
	    }
	}
    }
  edm::LogInfo("MCMatchAnalyzer::findRecoTrackMatch()")<<"Done findRecoTrackMatch().";
  return recoTrackMatch;
}


//________________________________________________________________________
TrackingParticleRef McMatchMuonAnalyzer::findSimTrackMatch(const edm::RefToBase<Track>& recoTrack,
							   reco::RecoToSimCollection& reco2Sim, 
							   Int_t& nMatches,Float_t& fFracShared)
{
  // return the matched sim track
  edm::LogInfo("MCMatchAnalyzer::findSimTrackMatch()")<<"Finding sim track match ...";

  TrackingParticleRef simTrackMatch;

  if(reco2Sim.find(recoTrack) != reco2Sim.end())  
    {
      edm::LogInfo("McMatchMuonAnalyzer::matchRecoMuons()")<<"Found Track match !!!!";
      vector<pair<TrackingParticleRef, double> > simTracks = reco2Sim[recoTrack];
      
      for(vector<pair<TrackingParticleRef, double> >::const_iterator it = simTracks.begin(); it != simTracks.end(); ++it)
	{
	  
	  TrackingParticleRef simTrack = it->first;
	  float fraction = it->second;
	  
	  if(fraction > matchhitfraction)
	    { 
	      simTrackMatch = simTrack; 
	      nMatches++; 
	      fFracShared = fraction;
	    }
	}
    }

  edm::LogInfo("MCMatchAnalyzer::findSimTrackMatch()")<<"Done finding sim track match!";

  return simTrackMatch;
}


//____________________________________________________________________________
Int_t McMatchMuonAnalyzer::getSimParentId(const TrackingParticleRef& match)
{
  // return the particle ID of associated GEN track (trackingparticle = gen + sim(geant) )
  // it is a final, stable particle (status=1): after final state radiaiton

  // same particle before final state radiation (status=3); from this one get to the real parent;
  // this is the 'parent' of the final state particle, p; same pdg_id

  // eg: look for the parent in the chain pion->mu+x
  // first parent, will be still the muon, before the final state radiation: pion->mu*+x->mu+x
  // from this parent get the parent that produced the muon, the pion in this case
  
  Int_t parentId = -99; 
  
  // if( match.isNull() ) return parentId;
  //  
  // edm::LogInfo("McMatchMuonAnalyzer::getSimParentId")<<"Getting the parent Id for the sim particle with status..."<< match->status();
  // for (TrackingParticle::genp_iterator b = match->genParticle_begin(); b != match->genParticle_end(); ++b)
  //   {
  //     const HepMC::GenParticle *mother = (*b)->production_vertex() && 
  //  (*b)->production_vertex()->particles_in_const_begin() != 
  //  (*b)->production_vertex()->particles_in_const_end() ?
  //  *((*b)->production_vertex()->particles_in_const_begin()):0;
  //     
  //     if( mother!=0 &&
  //    !(std::isnan(mother->momentum().m())) && !std::isnan(abs(mother->pdg_id())) &&
  //    abs(mother->pdg_id())<1e+6 )
  //  {
  //    if( abs(mother->pdg_id())<10 || (mother->pdg_id()<=100 && mother->pdg_id()>=80) ) return 0; // from PV (parton or cluster)
  //    else
  //      {
  //        parentId = mother->pdg_id();
  //    
  //        edm::LogInfo("McMatchMuonAnalyzer::getSimParentId")<<" 1 parentId = "<<parentId<<"\t parent_status= "<<mother->status();
  //        edm::LogInfo("McMatchMuonAnalyzer::getSimParentId")<<" 1 kidId = "<<match->pdgId()<<"\t kid_status= "<<match->status(); 
  //    
  //        // real parent after final state radiation
  //        const HepMC::GenParticle *motherMother = mother->production_vertex() &&
  //     mother->production_vertex()->particles_in_const_begin() !=
  //     mother->production_vertex()->particles_in_const_end() ?
  //     *(mother->production_vertex()->particles_in_const_begin()) : 0 ;
  //        
  //        if( motherMother!=0  
  //       && !(std::isnan(motherMother->momentum().perp()) ) && !(std::isnan(motherMother->pdg_id()) ) 
  //       && (motherMother->pdg_id() < 1e+6)
  //       )
  //     { 
  //       if(abs(motherMother->pdg_id())<10 || (motherMother->pdg_id()<=100 && motherMother->pdg_id()>=80))
  //         {
  //           if(parentId == match->pdgId()) return 0; // primary->X'->X
  //           else return parentId;
  //         }
  //       else
  //         {
  //           parentId = motherMother->pdg_id();
  //           edm::LogInfo("McMatchMuonAnalyzer::getSimParentId")<<" 2 parentId = "<<parentId<<"\t parent_status= "<<motherMother->status();
  //         }
  //       motherMother = 0;
  //     }// valid motherMother
  //      }//else: mother not from PV
  //  }//valid mother 
  //     mother = 0;
  //   }//loop over tracking particles
  //   
  // edm::LogInfo("McMatchMuonAnalyzer::getSimParentId")<<"Done with getSimParentId!\t"<<parentId;
  return parentId;
}


//________________________________________________________________________
Int_t McMatchMuonAnalyzer::getDetLayerId(const PSimHit& simHit)
{
  // return the detector layer ID
  LogDebug("MCMatchAnalyzer::getDetLayerId()")<<"Get detector layer ID ...";
 
  Int_t layerId;
  unsigned int id = simHit.detUnitId();

  if(theTracker->idToDetUnit(id)->subDetector() == GeomDetEnumerators::PixelBarrel)
  {
    PXBDetId pid(id);
    layerId = pid.layer() - 1;
  }

    else
  {
    PXFDetId pid(id);
    layerId = 2 + pid.disk();
  }
 
  LogDebug("MCMatchAnalyzer::getDetLayerId()")<<"Done";
  return layerId;
}


//___________________________________________________________________________
Int_t McMatchMuonAnalyzer::getNumberOfPixelHits(const TrackingParticleRef& simTrack_)
{
  // return number of pixel hits
  // pixel traker: 3 barel layers + 2 endcap disks
  LogDebug("MCMatchAnalyzer::getNumberOfPixelHits()")<<"Get number of pixel hits ...";

  // TrackingParticle *simTrack = const_cast<TrackingParticle *>(&simTrack_);
  std::vector<PSimHit> trackerPSimHit(getPSimHits(simTrack_, DetId::Tracker));

  const Int_t nLayers = 5;
  vector<bool> filled(nLayers,false);
  
  for(std::vector<PSimHit>::const_iterator
        simHit = trackerPSimHit.begin();
        simHit!= trackerPSimHit.end(); simHit++)
  {

    if(simHit == trackerPSimHit.begin())
      if(simHit->particleType() != simTrack_->pdgId())
	return 0;

    unsigned int id = simHit->detUnitId();

    if(theTracker->idToDet(id) && 
       (theTracker->idToDetUnit(id)->subDetector() == GeomDetEnumerators::PixelBarrel ||
	theTracker->idToDetUnit(id)->subDetector() == GeomDetEnumerators::PixelEndcap) ) 
      filled[getDetLayerId(*simHit)] = true;
  }
  
  // Count the number of filled pixel layers
  Int_t fLayers = 0;
  for(Int_t i=0; i<nLayers; i++)
    if(filled[i] == true) fLayers++;
   
  LogDebug("MCMatchAnalyzer::getNumberOfPixelLayers()")<<"Done!";

  return fLayers;
}


//______________________________________________________________________________
float McMatchMuonAnalyzer::refitWithVertex(const Track& recTrack)
{ 
  // returns the momentum of the track updated at th primary vertex
  LogDebug("MCMatchAnalyzer::refitWithVertex()")<<"Refit with primary vertex ...";
 
  TransientTrack theTransientTrack = theTTBuilder->build(recTrack);
  
  // If there are vertices found
  if(vertices->size() > 0)
    { 
      float dzmin = -1.; 
      const reco::Vertex * closestVertex = 0;
      
      // Look for the closest vertex in z
      for(reco::VertexCollection::const_iterator
	    vertex = vertices->begin(); vertex!= vertices->end(); vertex++)
	{
	  float dz = fabs(recTrack.vertex().z() - vertex->position().z());
	  if(vertex == vertices->begin() || dz < dzmin)
	    { dzmin = dz ; closestVertex = &(*vertex); }
	}
      
      // Get vertex position and error matrix
      GlobalPoint vertexPosition(closestVertex->position().x(),
				 closestVertex->position().y(),
				 closestVertex->position().z());
      
      float beamSize = theBeamSpot->BeamWidthX();
      GlobalError vertexError(beamSize*beamSize, 0,
			      beamSize*beamSize, 0,
			      0,closestVertex->covariance(2,2));
      
      // Refit track with vertex constraint
      SingleTrackVertexConstraint stvc;
      SingleTrackVertexConstraint::BTFtuple result =
	stvc.constrain(theTransientTrack, vertexPosition, vertexError);
      return result.get<1>().impactPointTSCP().pt();
    }
  else
    return recTrack.pt();
}


//_______________________________________________________________________
void McMatchMuonAnalyzer::dummVectorEntry(vector<float>& result, Int_t nEntries)
{
  // add to vector nentries of '-99' 
  for(Int_t i = 1; i<= nEntries; i++)
    result.push_back(-99);
}


//________________________________________________________________________
std::vector<PSimHit> McMatchMuonAnalyzer::getPSimHits(const TrackingParticleRef& st, DetId::Detector detector)
{
   std::vector<PSimHit> result;

   std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(st,TrackPSimHitRef());//SimHit is dummy: for simHitTPAssociationListGreater
                                                                                                  // sorting only the cluster is needed

   auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(),
         clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
   // TrackingParticle* simtrack = const_cast<TrackingParticle*>(&st);
   //loop over PSimHits
   // const PSimHit * psimhit=0;

   // look for the further most hit beyond a certain limit
   auto start=range.first;
   auto end=range.second;
   LogDebug("McMatchMuonAnalyzer")<<range.second-range.first<<" PSimHits.";

   // unsigned int count=0;
   for (auto ip=start;ip!=end;++ip){ 
      TrackPSimHitRef psit = ip->second;
      if ( detector == DetId( (uint32_t)(psit->detUnitId()) ).det() )
         result.push_back(*psit);
   }

   return result;
}

//________________________________________________________________________
std::vector<PSimHit> McMatchMuonAnalyzer::getPSimHits(const TrackingParticleRef& st)
{
   std::vector<PSimHit> result;

   std::pair<TrackingParticleRef, TrackPSimHitRef> clusterTPpairWithDummyTP(st,TrackPSimHitRef());//SimHit is dummy: for simHitTPAssociationListGreater
                                                                                                  // sorting only the cluster is needed

   auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(),
         clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
   // TrackingParticle* simtrack = const_cast<TrackingParticle*>(&st);
   //loop over PSimHits
   // const PSimHit * psimhit=0;

   // look for the further most hit beyond a certain limit
   auto start=range.first;
   auto end=range.second;
   LogDebug("McMatchMuonAnalyzer")<<range.second-range.first<<" PSimHits.";

   // unsigned int count=0;
   for (auto ip=start;ip!=end;++ip){ 
      TrackPSimHitRef psit = ip->second;
      result.push_back(*psit);
   }

   return result;
}

DEFINE_FWK_MODULE(McMatchMuonAnalyzer);
