/*  Implementation: 
0. Analyzer that gets as input 2 recoTrack collection and their respectiv sim2reco and reco2sim maps
1. makes single and pair track analysis
2. output: TNtuple

*/


// framework includes
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

// data formats
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoHI/HiCentralityAlgos/interface/CentralityProvider.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// sim data formats
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h" 
#include "HepMC/GenParticle.h"
#include "HepMC/GenVertex.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIBDetId.h" 
#include "DataFormats/SiStripDetId/interface/TIDDetId.h"
#include "DataFormats/SiStripDetId/interface/TOBDetId.h" 
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"


#include "SimMuon/MCTruth/interface/MuonAssociatorByHits.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"
#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

// ROOT
#include "TROOT.h"
#include "TNtuple.h"
#include "TLorentzVector.h"

// miscellaneous  
#include <fstream>
using namespace std;
using namespace reco;
const double m_mu    = 0.105658;
const double epsilon = 0.001;
//__________________________________________________________________________
class McMatchTrackAnalyzer : public edm::EDAnalyzer
{

public:
  explicit McMatchTrackAnalyzer(const edm::ParameterSet& pset);
  ~McMatchTrackAnalyzer();
  
  virtual void analyze(const edm::Event& ev, const edm::EventSetup& es);
  virtual void beginJob();
  virtual void endJob();
  
private:
  void fillEventInfo(vector<float>& result);
  void fillPairTrackTuple(const edm::RefToBase<Track>& kidRef1,
			  const edm::RefToBase<Track>& kidRef2,
			  vector<float>& result);
  void fillRecoPairTrackTuple(const edm::RefToBase<Track>& trkRef,
			      reco::RecoToSimCollection& reco2Sim,
			      vector<float>& result);
  void fillMatchedRecoTrackTuple(const edm::RefToBase<Track>& trkRef,
				 int& nmatches,
				 double fFrac,
				 vector<float>& result);
  void fillMatchedSimTrackTuple(const TrackingParticleRef& trkRef,
				int& nmatches,
				double fFrac,
				vector<float>& result);
  void fillRecoTrackTuple(const edm::RefToBase<Track>& trkRef,
			  vector<float>& result);
  void fillSimTrackTuple(const TrackingParticleRef& simTrack,
			 vector<float>& result);

  void fillSimHitsInfo(const TrackingParticleRef& simTrack,
		       vector<float>& result);
  void fillRecoHitsInfo(const edm::RefToBase<reco::Track>& trkRef, 
			vector<float>& result);
  
  edm::RefToBase<Track> findRecoTrackMatch(const TrackingParticleRef& trk,
					   reco::SimToRecoCollection& reco2Sim,
					   int& nmatches, 
					   double& fFracShared);
  TrackingParticleRef   findSimTrackMatch(const edm::RefToBase<Track>& trkRef,
					  reco::RecoToSimCollection& reco2Sim,
					  int& nmatches,
					  double& fFracShared);
  
  const reco::Vertex* getClosestVertex(edm::RefToBase<reco::Track>);
  int  getSimParentId(const TrackingParticleRef& trk);

  void matchRecoPairTracks(edm::Handle<edm::View<Track> >& trackCollection,
			   reco::RecoToSimCollection& p,	
			   TNtuple* tuple);
  void matchSimPairTracks(edm::Handle<TrackingParticleCollection>& simCollection,	
			  edm::Handle<edm::HepMCProduct>& hepEvt,	 
			  reco::SimToRecoCollection& trks, 
			  TNtuple* tuple); 
  
  void matchRecoTracks(edm::Handle<edm::View<Track> >& trackCollection,
		       reco::RecoToSimCollection& p,
		       TNtuple* tuple);
  void matchSimTracks(edm::Handle<TrackingParticleCollection>& simCollection,
		      reco::SimToRecoCollection& q,
		      TNtuple* tuple); 
  
  void dummVectorEntry(vector<float>& result, int entries);

  std::vector<PSimHit> getPSimHits(const TrackingParticleRef& st, DetId::Detector);
  std::vector<PSimHit> getPSimHits(const TrackingParticleRef& st);

  // ----- member data -----
  bool                         dohiembedding;
  bool                         doparticlegun;
  bool                         doreco2sim;
  bool                         dosim2reco;
  bool                         matchpair;
  bool                         matchsingle;
  int                          proc,procsgn,ntrk,nvtx;
  double                       b;
  int                          npart,ncoll,nhard;
  int                          pdgpair;
  int                          pdgsingle;
  
  TNtuple                     *pnRecoSimPairType1;
  TNtuple                     *pnRecoSimPairType2;
  TNtuple                     *pnRecoSimTrkType1;
  TNtuple                     *pnRecoSimTrkType2;
  TNtuple                     *pnSimRecoPairType1;
  TNtuple                     *pnSimRecoPairType2;
  TNtuple                     *pnSimRecoTrkType1;
  TNtuple                     *pnSimRecoTrkType2;
  TNtuple                     *pnEventInfo;

  const reco::VertexCollection *vertices;
  double                       vx, vy, vz;

  const reco::BeamSpot        *theBeamSpot;
  double                       bx,by,bz;
  // centrality
  CentralityProvider*          centrality;
  int                          centBin;
  int                          theCentralityBin;

  const SimHitTPAssociationProducer::SimHitTPAssociationList *simHitsTPAssoc;

  edm::InputTag                type1trktag;
  edm::InputTag                type1maptag;
  edm::InputTag                type2trktag;
  edm::InputTag                type2maptag;

  edm::InputTag                simtracktag;
  edm::InputTag                trktracktag; 
  edm::InputTag                trkmaptag;
  edm::InputTag                simHitTpMapTag; 
  edm::InputTag                vtxtag; 
};


//___________________________________________________________________________
McMatchTrackAnalyzer::McMatchTrackAnalyzer(const edm::ParameterSet& pset):
  dohiembedding(pset.getParameter<bool>("doHiEmbedding") ),
  doparticlegun(pset.getParameter<bool>("doParticleGun")),
  doreco2sim(pset.getParameter<bool>("doReco2Sim") ),
  dosim2reco(pset.getParameter<bool>("doSim2Reco") ),
  matchpair(pset.getParameter<bool>("matchPair") ),
  matchsingle(pset.getParameter<bool>("matchSingle") ),
  pdgpair(pset.getParameter<int>("pdgPair") ),
  pdgsingle(pset.getParameter<int>("pdgSingle") ),
  type1trktag(pset.getUntrackedParameter<edm::InputTag>("type1Tracks") ),
  type1maptag(pset.getUntrackedParameter<edm::InputTag>("type1MapTag") ),
  type2trktag(pset.getUntrackedParameter<edm::InputTag>("type2Tracks") ),
  type2maptag(pset.getUntrackedParameter<edm::InputTag>("type2MapTag") ),
  simtracktag(pset.getUntrackedParameter<edm::InputTag>("simTracks") ),
  simHitTpMapTag(pset.getUntrackedParameter<edm::InputTag>("tagForHitTpMatching") ),
  vtxtag(pset.getUntrackedParameter<edm::InputTag>("verticesTag") )
{
  // constructor
  centrality = 0;
  vx = 0.;
  vy = 0.;
  vz = 0.;
}


//_____________________________________________________________________________
McMatchTrackAnalyzer::~McMatchTrackAnalyzer()
{
  // destructor

  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//_____________________________________________________________________
void McMatchTrackAnalyzer::analyze(const edm::Event& ev, const edm::EventSetup& es)
{
  // method called each event  
 
  edm::LogInfo("MCMatchAnalyzer")<<"Start analyzing each event ...";

  // ---------Get generated event (signal or background)
  edm::Handle<edm::HepMCProduct> hepEvSgn, hepEv;
  ev.getByLabel("generator",hepEv);
  bool hepeventin =  ev.getByLabel("generator",hepEv);
  if(!hepeventin)  edm::LogInfo("McMatchTrackAnalyzer::analyze()")<<"NO HEPMC event! FIX IT!";

  proc = hepEv->GetEvent()->signal_process_id();
  edm::LogInfo("McMatchTrackAnalyzer::analyze()")<<"Process ID= " <<proc;

  // -------- embedded signal
  procsgn=0;
  if(dohiembedding)
    {
      ev.getByLabel("hiSignal",hepEvSgn);
      procsgn = hepEvSgn->GetEvent()->signal_process_id();
      edm::LogInfo("McMatchTrackAnalyzer::analyze()")<<"Process ID= " << procsgn;
    }

 const HepMC::HeavyIon* hi = hepEv->GetEvent()->heavy_ion();
 if(hi!=NULL)
   {
     ncoll = hi->Ncoll();
     nhard = hi->Ncoll_hard();
     if( hi->Npart_proj() + hi->Npart_targ() > 0)
       {
	 npart =  hi->Npart_proj() + hi->Npart_targ();
	 b = hi->impact_parameter();
       }
   }
 else
   {
     ncoll = 0;
     nhard = 0;
     npart = 0;
     b = -99.;  
   }

 // ----------- centrality
  // if(!centrality) centrality = new CentralityProvider(es);
  // centrality->newEvent(ev,es); // make sure you do this first in every event
  // centBin = centrality->getBin();
  centBin = 1;
 
  // --------- sim tracks
  edm::Handle<TrackingParticleCollection> simCollection;
  ev.getByLabel(simtracktag,simCollection);
  edm::LogInfo("McMatchTrackAnalyzer::analyze()")<< "Size of simCollection = " << simCollection.product()->size();  

  // ----------- vertex collection
  nvtx = 0;
  edm::Handle<reco::VertexCollection> vertexCollection;
  edm::LogInfo("McMatchTrackAnalyzer::analyze()") <<"Getting reco::VertexCollection - "<<vtxtag;
  bool vertexAvailable =  ev.getByLabel(vtxtag,vertexCollection);
  if (!vertexAvailable)
    {
      ev.getByLabel("pixelVertices",vertexCollection);
      edm::LogInfo("McMatchAnalyzer::analyze()")<<"Using the pp vertex collection: pixelVertices ";
    }
   
  nvtx     = vertexCollection.product()->size();
  vertices = vertexCollection.product();
  edm::LogInfo("McMatchAnalyzer::analyze()")<<"Number of vertices in the event = "<< nvtx;

  reco::VertexCollection::const_iterator vertex;
  if ( vertices->begin() != vertices->end() ) 
    {
      vertex = vertices->begin();
      vx = vertex->position().x();
      vy = vertex->position().y();
      vz = vertex->position().z();
    } 
  // ----------- beam spot
  edm::Handle<reco::BeamSpot>      beamSpotHandle;
  ev.getByLabel("offlineBeamSpot", beamSpotHandle);
  if(beamSpotHandle.isValid())
    {
      theBeamSpot = beamSpotHandle.product();
      bx = theBeamSpot->position().x();
      by = theBeamSpot->position().y();
      bz = theBeamSpot->position().z();
    }
 

  //get sim hits - TP map
  edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssocHandle;
  ev.getByLabel(simHitTpMapTag,simHitsTPAssocHandle);
  simHitsTPAssoc = simHitsTPAssocHandle.product();

  //-------
  int ntrk1 = 0;
  int ntrk2 = 0;
  edm::Handle<edm::View<reco::Track> >  type1TrkCollection;
  bool type1Available = ev.getByLabel(type1trktag, type1TrkCollection); 
  if (type1Available)
    {
      ev.getByLabel(type1trktag, type1TrkCollection); 
      ntrk1 = type1TrkCollection.product()->size();
      edm::LogInfo("McMatchTrackAnalyzer::analyze()")<< "Size of type 1 Collection = " << type1TrkCollection.product()->size();
    }
  else 
    edm::LogWarning("McMatchTrackAnalyzer::analyze()")<< "No type 1 tracks reconstructed ";

  edm::Handle<edm::View<reco::Track> >  type2TrkCollection;
  bool type2Available = ev.getByLabel(type2trktag, type2TrkCollection); 
  if (type2Available)
   {
     ev.getByLabel(type2trktag, type2TrkCollection); 
     ntrk2 = type2TrkCollection.product()->size();
     edm::LogInfo("McMatchTrackAnalyzer::analyze()")<< "Size of type 2 Collection = " << type2TrkCollection.product()->size();
   }
  else
    edm::LogWarning("McMatchTrackAnalyzer::analyze()")<< "No type 2 tracks reconstructed ";
   

  // --------- event info TNtuple 

  int nsimtrk = simCollection.product()->size();
  vector<float> result;
  result.push_back(proc); // proc
  result.push_back(procsgn); // proc signal
  result.push_back(nsimtrk); // ntrkr
  result.push_back(ntrk1); // ntrkr1
  result.push_back(ntrk2); // ntrkr2
  result.push_back(nvtx); // nvtxr
  result.push_back(ncoll);//ncoll
  result.push_back(nhard);//ncoll_hard
  result.push_back(npart);//npart
  result.push_back(b);//b
  fillEventInfo(result);
  pnEventInfo->Fill(&result[0]);
  result.clear();
 
  edm::Handle<reco::SimToRecoCollection> type1SimRecoHandle;     
  ev.getByLabel(type1maptag,type1SimRecoHandle);

  edm::Handle<reco::SimToRecoCollection> type2SimRecoHandle;
  ev.getByLabel(type2maptag,type2SimRecoHandle);
  //-------------- SIM2RECO
  if(dosim2reco)
    {
    
      if(type1SimRecoHandle.isValid() && type1Available)
	{
	  reco::SimToRecoCollection type1Sim2Reco = *(type1SimRecoHandle.product());
	  edm::LogInfo("McMatchTrackAnalyzer::analyze()")<<"Got sim2reco maps! Size ="<<type1Sim2Reco.size();
	  
	  if(matchsingle)
	    {
	      LogDebug("McMatchTrackAnalyzer::analyze()") <<"##### Matching single typeTracks sim2reco !!!!! ";
	      matchSimTracks(simCollection,type1Sim2Reco,pnSimRecoTrkType1);
	    }
	  
	  if(matchpair) 
	    {
	      if(dohiembedding)
		{ 
		  LogDebug("McMatchTrackAnalyzer::analyze()") <<"##### Matching pair typeTracks sim2reco !!!!! ";
		  matchSimPairTracks(simCollection,hepEvSgn,type1Sim2Reco,pnSimRecoPairType1);
		}
	      else 
		{
		  LogDebug("McMatchTrackAnalyzer::analyze()") <<"##### Matching pair typeTracks sim2reco !!!!! ";
		  matchSimPairTracks(simCollection,hepEv,type1Sim2Reco,pnSimRecoPairType1);
		}
	    } 
	}
      else LogDebug("McMatchTrackAnalyzer::analyze()") <<"##### No pre-existent sim2reco maps for type1 tracks!!!!! ";
      
     
      if(type2SimRecoHandle.isValid()&& type2Available)
	{
	  reco::SimToRecoCollection type2Sim2Reco = *(type2SimRecoHandle.product());
	  edm::LogInfo("McMatchTrackAnalyzer::analyze()")<<"Got sim2reco maps! Size ="<<type2Sim2Reco.size();
	  
	  if(matchsingle)
	    {
	      LogDebug("McMatchTrackAnalyzer::analyze()") <<"##### Matching single typeTracks sim2reco !!!!! ";
	      matchSimTracks(simCollection,type2Sim2Reco,pnSimRecoTrkType2);
	    }
	  
	  if(matchpair) 
	    {
	      LogDebug("McMatchTrackAnalyzer::analyze()") <<"##### Matching pair typeTracks sim2reco !!!!! ";
	      if(dohiembedding)
		matchSimPairTracks(simCollection,hepEvSgn,type2Sim2Reco,pnSimRecoPairType2);
	      else 
		matchSimPairTracks(simCollection,hepEv,type2Sim2Reco,pnSimRecoPairType2);
	    } 
	}
      else LogDebug("McMatchTrackAnalyzer::analyze()") <<"##### No pre-existent sim2reco maps for type2 tracks!!!!! ";
    } // dosim2reco


  //-----------------------------------------------
  // reco2sim
  if(doreco2sim)
    {
      // ##### type 1
      edm::Handle<reco::RecoToSimCollection> type1RecoSimHandle;
      ev.getByLabel(type1maptag,type1RecoSimHandle);
      if(type1RecoSimHandle.isValid() && type1Available)
	{
	  reco::RecoToSimCollection type1Reco2Sim = *(type1RecoSimHandle.product());
	  edm::LogInfo("McMatchTrackAnalyzer::analyze()")<<"Got reco2sim maps for type1 Size ="<<type1Reco2Sim.size();
	  
	  if(matchsingle) 
	    {
	      edm::LogInfo("McMatchTrackAnalyzer::analyze()") <<"##### Matching single typeTracks reco2Sim !!!!! ";
	      matchRecoTracks(type1TrkCollection,type1Reco2Sim,pnRecoSimTrkType1);
	    }
	  // pair analysis
	  if( matchpair && type1TrkCollection.product()->size() > 1 )
	    matchRecoPairTracks(type1TrkCollection,type1Reco2Sim,pnRecoSimPairType1);
	  else 
	    edm::LogInfo("McMatchTrackAnalyzer::analyze()") <<"##### Size of type 1 Reco2Sim <2 || !matchpair";
	} // valid tyeps handles
      else edm::LogInfo("McMatchTrackAnalyzer::analyze()") <<"##### No pre-existent reco2sim map or reco collection for type1 tracks!!!!! ";

      // ##### type2      
      edm::Handle<reco::RecoToSimCollection> type2RecoSimHandle;
      ev.getByLabel(type2maptag,type2RecoSimHandle);
      if(type2RecoSimHandle.isValid() && type2Available)
	{
	  reco::RecoToSimCollection type2Reco2Sim = *(type2RecoSimHandle.product());
	  edm::LogInfo("McMatchTrackAnalyzer::analyze()")<<"Got reco2sim maps for type2! Size ="<<type2Reco2Sim.size();
	  
	  if(matchsingle) 
	    {
	      edm::LogInfo("McMatchTrackAnalyzer::analyze()") <<"##### Matching single typeTracks reco2Sim !!!!! ";
	      matchRecoTracks(type2TrkCollection,type2Reco2Sim,pnRecoSimTrkType2);
	    }
	  // pair analysis
	  if( matchpair && type2TrkCollection.product()->size() > 1 )
	    matchRecoPairTracks(type2TrkCollection,type2Reco2Sim,pnRecoSimPairType2);
	  else 
	    edm::LogInfo("McMatchTrackAnalyzer::analyze()") <<"##### Size of type 2 Reco2Sim <2 || !matchpair";
	} // valid tyeps handles
      else edm::LogInfo("McMatchTrackAnalyzer::analyze()") <<"##### No pre-existent reco2sim map or reco collection for type2 tracks!!!!! ";
      
    }//doreco2sim
  
  edm::LogInfo("McMatchTrackAnalyzer::analyze()") << "[McMatchTrackAnalyzer] done with event# " << ev.id();
}


//________________________________________________________________________
void McMatchTrackAnalyzer::matchRecoTracks(edm::Handle<edm::View<reco::Track> >& trackCollection,
					   reco::RecoToSimCollection& p,
					   TNtuple* tuple)
{
  // fill the TNtuple pnRecoSimTracks, 
  // with reco tracks and their corresponding sim tracks information

  edm::LogInfo("MCMatchAnalyzer::matchRecoTracks()")<<"Start matching reco tracks ...";

  for(edm::View<reco::Track> ::size_type i=0; i < trackCollection.product()->size(); ++i)
  {
    edm::RefToBase<reco::Track> recTrack(trackCollection, i);
    if ( recTrack.isNull() ) continue;
    vector<float> result2;
   
    int nSim   = 0;
    double fFr = 0.;
    // reco tracker
    fillRecoTrackTuple(recTrack,result2);
    fillRecoHitsInfo(recTrack,result2);

    TrackingParticleRef matchedSimTrack = findSimTrackMatch(recTrack,p,nSim,fFr);
    fillMatchedSimTrackTuple(matchedSimTrack,nSim,fFr,result2);
    if(nSim>0)
      fillSimHitsInfo(matchedSimTrack,result2);  // 8 fields
    else dummVectorEntry(result2,8);
    fillEventInfo(result2);
    tuple->Fill(&result2[0]);
    result2.clear();
  }//i trackCollection
  edm::LogInfo("MCMatchAnalyzer::matchRecoTracks()")<<"Done matchRecoTracks().";
}


//_________________________________________________________________________
void McMatchTrackAnalyzer::matchSimTracks(edm::Handle<TrackingParticleCollection>& simCollection,
					  reco::SimToRecoCollection& q,
					  TNtuple* tuple)
{
  // fill TNtuple pnSimRecoTracks with sim track and corresponding reco tracks info
  // what are the trackingParticles with status = -99? 
  // are they from the decay in GEANT of the original particles? 
  // yes, they are the particles with no correspondence in the genPArticelCollection ... so if doing specific particle analysis, i should require only the 'signal' muons, not the decay one ...

  edm::LogInfo("MCMatchAnalyzer::matchSimTracks()")<<"Start matching simTracks ...";
 
  for(TrackingParticleCollection::size_type i=0; i < simCollection.product()->size(); i++)
    {
      const TrackingParticleRef simTrack(simCollection,i);
      if( simTrack.isNull() ) continue;

      // only muons
      if( (abs(simTrack->pdgId()) != pdgsingle) ) continue;
      if( simTrack->charge() == 0 ) continue;

      // discard very low pt muons, which have no chance to be reconstructed but increase the output ntuple size like hell
      if (simTrack->pt()<1) continue;
  
      vector<float> result4;

      // sim track
      fillSimTrackTuple(simTrack,result4); // 12 fields
      fillSimHitsInfo(simTrack,result4);  // 8 fields
      // recoTracker matched 
      int nRec = 0;
      double fF = 0.;
      edm::RefToBase<reco::Track> matchedRecTrack = findRecoTrackMatch(simTrack,q,nRec,fF);

      fillMatchedRecoTrackTuple(matchedRecTrack,nRec,fF,result4); //
      fillRecoHitsInfo(matchedRecTrack,result4);            //
      // fill
      fillEventInfo(result4);
      tuple->Fill(&result4[0]);

      result4.clear();
    }

  edm::LogInfo("MCMatchAnalyzer::matchSimTracks()")<<"Done matchSimTracks()!";
}


//______________________________________________
void McMatchTrackAnalyzer::matchRecoPairTracks(edm::Handle<edm::View<reco::Track> >& trkTypeCollection,
					       reco::RecoToSimCollection& trkReco2Sim, 
					       TNtuple* tuple)
{
  // fill the TNtuple pnRecoSimPairTracks, 
  // with reco and corresponding sim tracks information

  edm::LogInfo("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"Start reconstructing PairTracks!";

  for(edm::View<reco::Track> ::size_type i1=0; i1 < trkTypeCollection.product()->size()-1; i1++)
    {
      edm::RefToBase<reco::Track> trk1(trkTypeCollection, i1);
      if( trk1.isNull() ) continue;
      for(edm::View<reco::Track> ::size_type i2=i1+1; i2 < trkTypeCollection.product()->size(); i2++)
      	{
	  edm::RefToBase<reco::Track> trk2(trkTypeCollection, i2);
	  if( trk2.isNull() ) continue;

	  if(trk1->charge()*trk2->charge()>0) continue; // just oppposite sign pairs
	  
	  vector<float> result;
	  // fill pair with global tracks as kids
	  fillPairTrackTuple(trk1,trk2,result);// 4 fields
		  
	  // first muon  
	  fillRecoTrackTuple(trk1,result); // 11 fields
	  int nSim  = 0;
	  double fFr = 0.;
	  // matched sim track
	  TrackingParticleRef matchedSimTrack1 = findSimTrackMatch(trk1,trkReco2Sim,nSim,fFr);
	  fillMatchedSimTrackTuple(matchedSimTrack1,nSim,fFr,result); // 8 fields
	  LogDebug("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"First kid field filled!";

	  // second muon
	  fillRecoTrackTuple(trk2,result); // 11 fields
	  LogDebug("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"Second kid field filled!";
	  TrackingParticleRef matchedSimTrack2 = findSimTrackMatch(trk2,trkReco2Sim,nSim,fFr);
	  fillMatchedSimTrackTuple(matchedSimTrack2,nSim,fFr,result);
	  LogDebug("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"First kid field filled!";
	  fillEventInfo(result);
	  tuple->Fill(&result[0]);

	  edm::LogInfo("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"SIZE = "<<(int)result.size();

	  LogDebug("McMatchTrackAnalyzer::matchRecoPairTracks():result")<<"eta mu1 = "<<result[4]<<"\t eta mu2 = "<< result[13];
	  LogDebug("McMatchTrackAnalyzer::matchRecoPairTracks():direct")<<"eta mu1 = " << trk1->eta() <<"\t ta mu2 = " << trk2->eta();
	  result.clear();
	}// second muon loop
    }// first muon loop
  edm::LogInfo("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"Done matchRecoPairTracks()!";
}


//_________________________________________________________________________
void McMatchTrackAnalyzer::matchSimPairTracks(edm::Handle<TrackingParticleCollection>& simCollection,
					      edm::Handle<edm::HepMCProduct>& evt,
					      reco::SimToRecoCollection& trkSim2Reco,
					      TNtuple* tuple)
{
  // fill TNtuple pnSimRecoPairTracks with gen info for the gen Z0 and muon daughters and reco info for the reco muons

  // status == 1 : stable particle at pythia level, but which can be decayed by geant
  // status == 2 :
  // status == 3 : outgoing partons from hard scattering

  // The barcode is the particle's reference number, every vertex in the
  // event has a unique barcode. Particle barcodes are positive numbers,
  // vertex barcodes are negative numbers.

  edm::LogInfo("MCMatchAnalyzer::matchSimPairTracks()")<<"Start matching pairs ...";
  int statuscheck = 3;
  if(doparticlegun) statuscheck = 2;

  const HepMC::GenEvent * myGenEvent = evt->GetEvent();
  Bool_t foundmuons = false;
  vector<int> ixmu;
  vector<float> result3;
  edm::LogInfo("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"SIZE_1_result3 = "<<result3.size();

  //====================================
  for ( HepMC::GenEvent::particle_const_iterator p = myGenEvent->particles_begin(); p != myGenEvent->particles_end(); ++p ) 
    { 
      if ( ( abs((*p)->pdg_id()) == pdgpair ) && (*p)->status() == statuscheck ) 
	{ 
	  for( HepMC::GenVertex::particle_iterator aDaughter=(*p)->end_vertex()->particles_begin(HepMC::descendants); 
	       aDaughter !=(*p)->end_vertex()->particles_end(HepMC::descendants);
	       aDaughter++)
	    {
	      if ( abs((*aDaughter)->pdg_id())==pdgsingle && (*aDaughter)->status()==1 )
		{
		  ixmu.push_back((*aDaughter)->barcode());
		  edm::LogInfo("MCMatchAnalyzer::matchSimPairTracks()") << "Stable muon from Z0" << "\tindex= "<< (*aDaughter)->barcode(); 
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
   //==================================
  edm::LogInfo("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"SIZE_2_result3 = "<<result3.size();
  if( foundmuons )
    {
      vector<edm::RefToBase<reco::Track> > recokids;
      int nrecokids    = 0;
      int nsimtracks   = 0;
      for(TrackingParticleCollection::size_type i=0; i < simCollection.product()->size(); i++)
	{
	  const TrackingParticleRef simTrk(simCollection, i);
	  if( simTrk.isNull() || simTrk->status()!=1 ) continue;
	  const SimTrack *gTrack = &(*simTrk->g4Track_begin());
	  if( gTrack == NULL ) continue;

	  if(abs(gTrack->type())==pdgsingle && (gTrack->genpartIndex()==ixmu[0] || gTrack->genpartIndex()==ixmu[1] ))
	    {
	      edm::LogInfo("MCMatchAnalyzer::matchSimPairTracks()")<<"Got an xmuon!.";
	      nsimtracks++;
	      // sim track
	      fillSimTrackTuple(simTrk,result3); //10 fields
	      fillSimHitsInfo(simTrk,result3);  // 8 fields
	      // trackerMatched_muon		 
	      int nTrk = 0;
	      double fTrkHit = 0.; 
	      edm::RefToBase<reco::Track> matchedRecTrk = findRecoTrackMatch(simTrk,trkSim2Reco,nTrk,fTrkHit);
	      fillMatchedRecoTrackTuple(matchedRecTrk,nTrk,fTrkHit,result3);  // 14 fields
	      fillRecoHitsInfo(matchedRecTrk,result3);            // 22 fields
	       edm::LogInfo("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"SIZE_3_result3 = "<<result3.size();
	      if(nTrk!=0)
		{
		  recokids.push_back(matchedRecTrk);
		  nrecokids++;
		}
	    }// muon id
	}// TrackCollection loop
      //---------------
      // have the sim kids in recokids vector
      edm::LogInfo("MCMatchAnalyzer::matchSimPairTracks()")<<"Found kids: "<<nrecokids<<"SIZE_4_result3 = "<<result3.size();
      if(nsimtracks==0) 
	{
	  edm::LogInfo("MCMatchAnalyzer::matchSimPairTracks()")<<"No sim found for the gen kids";
	  dummVectorEntry(result3,108); //(2*25(sim+reco))
	}
      if(nsimtracks==1)
	{
	  edm::LogInfo("MCMatchAnalyzer::matchSimPairTracks()")<<"Only one sim out of the 2 kids were found";
	  dummVectorEntry(result3,54); //(1*25)
	  	  
	}
      if(nrecokids == 0 || nrecokids==1) // no or just one kid reconstructed; for the recoZ
	{
	  LogDebug("MCMatchAnalyzer::matchSimPairTracks()")<<"No sim daughter reconstructed";
	  dummVectorEntry(result3,4); //(4 recoZ0)
	}
     
      // get the reco Z0 from global muons
      if(nrecokids == 2)
	{
	  edm::LogInfo("MCMatchAnalyzer::matchSimPairTracks()")<<"########## Fill reco_pair and reco_kids info";
	  edm::RefToBase<reco::Track> ch0 = recokids[0];
	  edm::RefToBase<reco::Track> ch1 = recokids[1];
	  
	  // reconstructed Z0
	  fillPairTrackTuple(ch0,ch1,result3); // 4fields
	}

      edm::LogInfo("McMatchTrackAnalyzer::matchRecoPairTracks()")<<"SIZE_7_result3 = "<<result3.size();
      fillEventInfo(result3);
      tuple->Fill(&result3[0]);
      recokids.clear();
      result3.clear();
    }//foundmuons
 
  result3.clear();
 

  edm::LogInfo("MCMatchAnalyzer::matchSimPairTracks()")<<"Done matchSimPairTracks()!.";
}


//_________________________________________________________________________
void  McMatchTrackAnalyzer::fillEventInfo(vector<float>& result)
{
  // reconstruct the parent dilepton composed object: y,minv,phi,pT
  result.push_back(centBin);//centrality bin
  result.push_back(bx);//beam spot
  result.push_back(by);//beam spot
  result.push_back(bz);//beam spot
  result.push_back(vx);//vertex
  result.push_back(vy);//vertex
  result.push_back(vz);//vertex
 
  edm::LogInfo("MCMatchAnalyzer::fillEventInfo()")<<"Done filling event info stuff!";
}


//_________________________________________________________________________
void  McMatchTrackAnalyzer::fillPairTrackTuple(const edm::RefToBase<reco::Track>& kidRef1,
					       const edm::RefToBase<reco::Track>& kidRef2,
					       vector<float>& result)
{
  // reconstruct the parent dilepton composed object: y,minv,phi,pT

  TLorentzVector child1, child2;
  double en1 = sqrt(kidRef1->p()*kidRef1->p()+m_mu*m_mu);
  double en2 = sqrt(kidRef2->p()*kidRef2->p()+m_mu*m_mu);
  
  child1.SetPxPyPzE(kidRef1->px(),kidRef1->py(),kidRef1->pz(),en1);
  child2.SetPxPyPzE(kidRef2->px(),kidRef2->py(),kidRef2->pz(),en2);
  
  TLorentzVector pair;
  pair = child1 + child2;
	 	  

  result.push_back(pair.Rapidity());
  result.push_back(pair.M());
  result.push_back(pair.Phi());
  result.push_back(pair.Pt());

  edm::LogInfo("MCMatchAnalyzer::fillPairTrackTuple()")<<"Done fillPairTrackTuple()!";
}


//_______________________________________________________________________
void McMatchTrackAnalyzer::fillRecoPairTrackTuple(const edm::RefToBase<reco::Track>& trkRef,
						  reco::RecoToSimCollection& reco2Sim,
						  vector<float>& result)
{
  // get the match, 
  // fill the result with the id of match, id of parent, and and number of matches
  edm::LogInfo("MCMatchAnalyzer::fillRecoPairTrackTuple()")<<"Filling pair tuple ...";

  if(!trkRef.isNull())
    {
      int nTrkSim = 0;
      double fFraction = 0.;
      TrackingParticleRef matchedSim = findSimTrackMatch(trkRef,reco2Sim,nTrkSim,fFraction); 
      int nId = -99;
      int nIdParent = -99;
      
      if( !matchedSim.isNull() && nTrkSim!=0 )
	{
	  nId = matchedSim->pdgId();
	  nIdParent = getSimParentId(matchedSim);
	}
      
      result.push_back(nTrkSim);
      result.push_back(fFraction);
      result.push_back(nId);
      result.push_back(nIdParent);
    }
  else dummVectorEntry(result,4);
  edm::LogInfo("MCMatchAnalyzer::fillRecoPairTrackTuple()")<<"Done fillRecoPairTrackTuple()!";
 }


//____________________________________________________________________
void McMatchTrackAnalyzer::fillMatchedRecoTrackTuple(const edm::RefToBase<reco::Track>& recoTrack,
						int& nMatches, double fFracMatch,vector<float>& result)
{
  // the info part of the matched sim track
  edm::LogInfo("MCMatchAnalyzer::fillMatchedTrackTuple()")<<"Filling matched reco track info ...";

  if(!recoTrack.isNull() && nMatches > 0)
    {
      double dt      = -99.;
      double sigmaDt = -99.;
      double dz      = -99.;
      double sigmaDz = -99. ;
      
      if(getClosestVertex(recoTrack)!=NULL)
	{
	  const reco::Vertex *vtx = getClosestVertex(recoTrack);
	  dt      = recoTrack->dxy(vtx->position());
	  sigmaDt = sqrt(recoTrack->dxyError()*recoTrack->dxyError() + 
			 vtx->yError()*vtx->yError()+vtx->xError()*vtx->xError());
	  dz      = recoTrack->dz(vtx->position());
	  sigmaDz = sqrt(recoTrack->dzError()*recoTrack->dzError()+vtx->zError()*vtx->zError());
	}
      
      result.push_back(nMatches);            // number of reco tracks matched to one sim track (split tracks)
      result.push_back(fFracMatch);          // fraction of hits matched, for the last matche found, the one kept
      result.push_back(recoTrack->charge()); // chargereco
      result.push_back(recoTrack->chi2());                 // normalized chi2
      result.push_back(recoTrack->normalizedChi2());       // n && getClosestVertex(recoTrack)!=NULLormalized Chi2
      result.push_back(dt);                                // transverse DCA
      result.push_back(sigmaDt);                           // Sigma_Dca_t
      result.push_back(dz);                                // z DCA
      result.push_back(sigmaDz);                           // Sigma_Dca_z
      result.push_back(recoTrack->numberOfValidHits());    // number of hits found 
      result.push_back(recoTrack->eta());    // eta
      result.push_back(recoTrack->phi());    // phireco
      result.push_back(recoTrack->pt());     // ptreco
      result.push_back(recoTrack->ptError());     // ptreco error
      result.push_back(recoTrack->algo());     // algo (step at which the track was found)
      // cout << __LINE__ << " " << recoTrack->algo() << " " << recoTrack->algoName() << endl;
      
      edm::LogInfo("MCMatchAnalyzer::fillMatchedSimTrackTuple()")<<"pt= "<<recoTrack->pt();
    }
  else dummVectorEntry(result,15);
  edm::LogInfo("MCMatchAnalyzer::fillMatchedTrackTuple()")<<"Done fillMatchedTrackTuple()!"; 
}


//____________________________________________________________________
void McMatchTrackAnalyzer::fillMatchedSimTrackTuple(const TrackingParticleRef& simTrack,
					       int& nMatches, double fFrac,vector<float>& result)
{
  // the info part of the matched sim track
  edm::LogInfo("MCMatchAnalyzer::fillMatchedTrackTuple()")<<"Filling matched sim track info ...";

  if( !simTrack.isNull() && nMatches>0)
    {
      
      int nId       = simTrack->pdgId();
      int nIdParent = getSimParentId(simTrack);
      edm::LogInfo("MCMatchAnalyzer::fillMatchedSimTrackTuple()")<<"pt= "<<simTrack->pt()<<"\t particle id = "<<nId<<"\t parent id = "<<nIdParent<<"\tstatus= "<<simTrack->status();

      result.push_back(nMatches);                 // number of sim tracks matched to one reco track
      result.push_back(fFrac);                    // fraction of hits matched, for the last matche found, the one kept
      result.push_back(simTrack->charge());       // charge sim
      result.push_back(simTrack->eta());          // eta sim
      result.push_back(nId);                      // id
      result.push_back(nIdParent);                // id parent
      result.push_back(simTrack->phi());          // phisim
      result.push_back(simTrack->pt());           // pt sim
    }
  else dummVectorEntry(result,8);
  edm::LogInfo("MCMatchAnalyzer::fillMatchedSimTrackTuple()")<<"Done fillMatchedTrackTuple()!"; 
}


//_______________________________________________________________________
void McMatchTrackAnalyzer::fillRecoTrackTuple(const edm::RefToBase<reco::Track>& trkRef, vector<float>& result)
{
  // fill reco track info to be fed later to the tuple
  edm::LogInfo("MCMatchAnalyzer::fillRecoTrackTuple()")<<"Filling reco track info ...";
 
 
  double dt      = -99.;
  double sigmaDt = -99.;
  double dz      = -99.;
  double sigmaDz = -99. ;
  
  if(getClosestVertex(trkRef)!=NULL)
    {
      const reco::Vertex *vtx = getClosestVertex(trkRef);
      dt      = trkRef->dxy(vtx->position());
      sigmaDt = sqrt(trkRef->dxyError()*trkRef->dxyError() + 
		     vtx->yError()*vtx->yError()+vtx->xError()*vtx->xError());
      dz      = trkRef->dz(vtx->position());
      sigmaDz = sqrt(trkRef->dzError()*trkRef->dzError()+vtx->zError()*vtx->zError());
    }
   
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
   result.push_back(trkRef->ptError());                        // pt
   result.push_back(trkRef->algo());                           // algo (step at which the track was found)
   // cout << __LINE__ << " " << trkRef->algo() << " " << trkRef->algoName() << endl;
   
   edm::LogInfo("MCMatchAnalyzer::fillRecoTrackTuple()")<<"Done fillRecoTrackTuple()!";
}


//____________________________________________________________________
void McMatchTrackAnalyzer::fillSimTrackTuple(const TrackingParticleRef& simTrack,
					     vector<float>& result)
{
  // the info part of the matched sim track
  edm::LogInfo("MCMatchAnalyzer::fillMatchedTrackTuple()")<<"Filling sim track info ...";

  int nParentId    = getSimParentId(simTrack);
  result.push_back(simTrack->charge());  // chargereco
  result.push_back(simTrack->vertex().Rho()); // dca
  result.push_back(simTrack->vertex().x()); // x of the vtx
  result.push_back(simTrack->vertex().y()); // y of the vtx
  result.push_back(simTrack->eta());     // eta
  result.push_back(simTrack->pdgId());   // pdg id
  result.push_back(nParentId);           // parent id
  result.push_back(simTrack->phi());     // phi
  result.push_back(simTrack->pt());      // pt
  result.push_back(simTrack->status());  // status
 
  edm::LogInfo("MCMatchAnalyzer::fillSimTrackTuple()")<<"Done fillSimTrackTuple()!"; 
}


//_____________________________________________________________
void McMatchTrackAnalyzer::fillSimHitsInfo(const TrackingParticleRef& simTrack,
					   vector<float>& result)
{
  // fill sim hits info
  edm::LogInfo("HitsAnalyzer")<<"Filling simHits info ...";
  /*
  "nhits:ntrkerhits:npixelhits:nsilhits:"  // sim track 8
  "nmuonhits:ndthits:ncschits:nrpchits:"
  */

  std::vector<PSimHit> trackerPSimHit(getPSimHits(simTrack));
  
      double simhitssize  = double((getPSimHits(simTrack)).size()); // hits from all detectors
      
      double simtrkerhits = double((getPSimHits(simTrack,DetId::Tracker)).size()); // tracker hits
      int simpixelhits = 0;
      int simstriphits = 0;
      
      double simmuonhits  = double((getPSimHits(simTrack,DetId::Muon)).size()); // muon hits
      int simdthits    = 0;
      int simcschits   = 0;
      int simrpchits   = 0;
      
      
      for(std::vector<PSimHit>::const_iterator simHit = trackerPSimHit.begin();
	  simHit!= trackerPSimHit.end(); simHit++)
	{
	  const DetId detId = DetId(simHit->detUnitId());
	  DetId::Detector detector = detId.det();
	  int subdetId = static_cast<int>(detId.subdetId());
	  
	  //  if(subdetId==PixelSubdetector::PixelBarrel || subdetId==PixelSubdetector::PixelEndcap) simpixelhits++;
	  if(detector == DetId::Tracker && (subdetId==PixelSubdetector::PixelBarrel || subdetId==PixelSubdetector::PixelEndcap) ) simpixelhits++;
	  
	  if (detector == DetId::Tracker && 
	      (subdetId==SiStripDetId::TIB||subdetId==SiStripDetId::TOB||
	       subdetId==SiStripDetId::TID||subdetId==SiStripDetId::TEC) ) simstriphits++;
	  if(detector == DetId::Muon && subdetId == MuonSubdetId::DT) simdthits++;
	  if(detector == DetId::Muon && subdetId == MuonSubdetId::CSC) simcschits++;
	  if(detector == DetId::Muon && subdetId == MuonSubdetId::RPC) simrpchits++;
	}
      
      result.push_back(simhitssize);
      
      result.push_back(simtrkerhits);
      result.push_back(simpixelhits);
      result.push_back(simstriphits);
      
      result.push_back(simmuonhits);
      result.push_back(simdthits);
      result.push_back(simcschits);
      result.push_back(simrpchits);
  
      edm::LogInfo("MCMatchAnalyzer::fillSimHitsInfo()")<<"Done fillSimHitsInfo()!"; 
}


//______________________________________________________________
void McMatchTrackAnalyzer::fillRecoHitsInfo(const edm::RefToBase<reco::Track>& trkRef, vector<float>& result)
{
  // fill reco track info to be fed later to the tuple
  edm::LogInfo("MCMatchAnalyzer::fillRecoTrackTuple()")<<"Filling reco hits info ...";
  /*
    "nhits:nvalidhits:"
    "nvalidtrkerhits:nvalidpixelhits:nvalidstriphits:"
    "nvalidmuonhits:nvaliddthits:nvalidcschits:nvalidrpchits:"
    "nlosthits:"
    "nlosttrkerhits:nlostpixelhits:nloststriphits:"
    "nlostmuonhits:nlostdthits:nlostcschits:nlostrpchits:"
    "nbadhits:"
    "nbadmuonhits:nbaddthits:nbadcschits:nbadrpchits"
    
  */
  //22
  if(!trkRef.isNull())
    {
      const reco::HitPattern& hp = trkRef.get()->hitPattern();

      result.push_back(hp.numberOfHits(reco::HitPattern::TRACK_HITS));
      result.push_back(hp.numberOfValidHits());
      
      result.push_back(hp.numberOfValidTrackerHits());
      result.push_back(hp.numberOfValidPixelHits());
      result.push_back(hp.numberOfValidStripHits());
      
      result.push_back(hp.numberOfValidMuonHits());
      result.push_back(hp.numberOfValidMuonDTHits());
      result.push_back(hp.numberOfValidMuonCSCHits());
      result.push_back(hp.numberOfValidMuonRPCHits());
      
      result.push_back(hp.numberOfLostHits(reco::HitPattern::TRACK_HITS));
      
      result.push_back(hp.numberOfLostTrackerHits(reco::HitPattern::TRACK_HITS));
      result.push_back(hp.numberOfLostPixelHits(reco::HitPattern::TRACK_HITS));
      result.push_back(hp.numberOfLostStripHits(reco::HitPattern::TRACK_HITS));
      
      result.push_back(hp.numberOfLostMuonHits());
      result.push_back(hp.numberOfLostMuonDTHits());
      result.push_back(hp.numberOfLostMuonCSCHits());
      result.push_back(hp.numberOfLostMuonRPCHits());
      
      result.push_back(hp.numberOfBadHits());
      result.push_back(hp.numberOfBadMuonHits());
      result.push_back(hp.numberOfBadMuonDTHits());
      result.push_back(hp.numberOfBadMuonCSCHits());
      result.push_back(hp.numberOfBadMuonRPCHits());
    }
  else dummVectorEntry(result,22); 
  edm::LogInfo("MCMatchAnalyzer::fillRecoHitsInfo()")<<"Done fillRecoHitsInfo()!"; 
}


//________________________________________________________________________
edm::RefToBase<reco::Track> McMatchTrackAnalyzer::findRecoTrackMatch(const TrackingParticleRef& simTrack,
								     reco::SimToRecoCollection& sim2Reco, 
								     int& nMatches, 
								     double& fFracShared)
{
  // return the matched reco track
  edm::LogInfo("MCMatchAnalyzer::findRecoTrackMatch()")<<"Finding reco track match ...";
  edm::LogInfo("McMatchTrackAnalyzer::findRecoTrackMatch()")<<" for simTrack with ... Pdg_ID = " << simTrack->pdgId()
							    <<"\t status = "<<simTrack->status()
							    <<"\t pt = "<<simTrack->pt()
							    <<"\t size of the collection: ="<<sim2Reco.size();
  edm::RefToBase<reco::Track> recoTrackMatch;
  if(sim2Reco.find(simTrack) != sim2Reco.end()) 	 
    {
      std::vector<std::pair<edm::RefToBase<reco::Track>,double> > recTracks = (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >)sim2Reco[simTrack];
      if(recTracks.size() !=0 )
	{
	  recoTrackMatch = recTracks.begin()->first;
	  fFracShared    = recTracks.begin()->second;
	  nMatches       = recTracks.size();
	}
    }

  edm::LogInfo("MCMatchAnalyzer::findRecoTrackMatch()")<<"Done findRecoTrackMatch(). Found "<<nMatches<<"\t matches";
  return recoTrackMatch;
}


//________________________________________________________________________
TrackingParticleRef McMatchTrackAnalyzer::findSimTrackMatch(const edm::RefToBase<reco::Track>& recoTrack,
							    reco::RecoToSimCollection& reco2Sim, 
							    int& nMatches,
							    double& fFracShared)
{
  // return the matched sim track
  edm::LogInfo("MCMatchAnalyzer::findSimTrackMatch()")<<"Finding sim track match ...";
  edm::LogInfo("McMatchTrackAnalyzer::findRecoTrackMatch()")<<" for recoTrack with ..pT  = " << recoTrack->pt()
							    <<"\t size of the collection: ="<<reco2Sim.size();
  TrackingParticleRef simTrackMatch;
 
  if(reco2Sim.find(recoTrack) != reco2Sim.end())  
    {
      vector<pair<TrackingParticleRef, double> > simTracks = reco2Sim[recoTrack];
      if(simTracks.size() !=0 )
      	{
	  simTrackMatch = simTracks.begin()->first;
	  fFracShared   = simTracks.begin()->second;
	  nMatches      = simTracks.size();
	}
    }

  edm::LogInfo("MCMatchAnalyzer::findSimTrackMatch()")<<"Done finding sim track match! Found "<<nMatches<<"matches";

  return simTrackMatch;
}


//___________________________________________________________________________
const reco::Vertex* McMatchTrackAnalyzer::getClosestVertex(edm::RefToBase<reco::Track> recTrack)
{
  // get the xloseset event vertex to the track
  const reco::Vertex *closestVertex=0;
  if(vertices->size()>0)
    {
      //look for th elcosest vertex in z

      double dzmin = -1;
      for(reco::VertexCollection::const_iterator vertex = vertices->begin(); vertex != vertices->end(); vertex++)
	{
	  double dz = fabs(recTrack->vertex().z() - vertex->position().z());
	  if(vertex == vertices->begin() || dz < dzmin)
	    {
	      dzmin = dz;
	      closestVertex = &(*vertex);
	    }
	}
    }
  else edm::LogInfo("MCMatchAnalyzer::getClossestVertex()")<<"Null vertex in the event";

  return closestVertex;

}


//____________________________________________________________________________
int McMatchTrackAnalyzer::getSimParentId(const TrackingParticleRef& match)
{
  // return the particle ID of associated GEN track (trackingparticle = gen + sim(geant) )
  // it is a final, stable particle (status=1): after final state radiaiton

  // same particle before final state radiation (status=3); from this one get to the real parent;
  // this is the 'parent' of the final state particle, p; same pdg_id
  
  int parentId = -99; 
  
  if( match.isNull() ) return parentId;
   
  edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Getting the parent Id for the sim particle with status..."<< match->status();
  TrackingParticle::genp_iterator b;
  TrackingParticle::genp_iterator in  = match->genParticle_begin();
  TrackingParticle::genp_iterator fin = match->genParticle_end();
  for (b = in; b != fin; ++b)
    {
      const reco::GenParticle *p = b->get();
      if( p==NULL ) 
	{
	  edm::LogInfo("MCMatchAnalyzer::getSimParentId()")<<"No gen particle associated with simTrack with status " << match->status();
	  continue;
	}
  
      edm::LogInfo("MCMatchAnalyzer::getSimParentId()")<<"Gen particle is : " << p->pdgId();
      const reco::Candidate *mother = p->numberOfMothers()>0 ?
	p->mother(0) : 0;
    
      if( mother!=0 && !(std::isnan(mother->pt())) && !std::isnan(abs(mother->pdgId())) )
	{
	  parentId = mother->pdgId();
	  if( parentId != p->pdgId() && !(parentId<=100 && parentId>=80) ) 
	    {
	      edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Id_parent = "<<parentId;
	      return parentId;
	    }
	  edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<" 1st parentId = "<<parentId<<"\t parent_status= "<<mother->status();

	  const reco::Candidate *motherMother = mother->numberOfMothers()>0 ?
	    mother->mother(0) : 0 ;	  
	  if( motherMother!=0 && !(std::isnan(motherMother->pt()) ) && !(std::isnan(motherMother->pdgId()) ) )
	    { 
	      parentId = motherMother->pdgId();
	      if( parentId != p->pdgId()  && !(parentId<=100 && parentId>=80) ) 
		{
		  edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Id_parent = "<<parentId;
		  return parentId;
		}
	      edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<" 2 parentId = "<<parentId<<"\t parent_status= "<<motherMother->status();
		      
	      // 3rd mother:
	      const reco::Candidate *motherMotherMother = motherMother->numberOfMothers()>0 ?
		motherMother->mother(0) : 0 ;
	      
	      if( motherMotherMother!=0 && !(std::isnan(motherMotherMother->pt()) ) && !(std::isnan(motherMotherMother->pdgId()) ) )
		{ 
		  parentId = motherMotherMother->pdgId();
		  if( parentId != p->pdgId()  && !(parentId<=100 && parentId>=80) ) 
		    {
		      edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Id_parent = "<<parentId;
		      return parentId;
		    }

		  // 4ht mother
		  const reco::Candidate *motherMotherMotherMother = motherMotherMother->numberOfMothers()>0 ?
		    motherMother->mother(0) : 0 ;
		  
		  if( motherMotherMotherMother!=0 && !(std::isnan(motherMotherMotherMother->pt()) ) && !(std::isnan(motherMotherMotherMother->pdgId()) ) )
		    { 
		      parentId = motherMotherMotherMother->pdgId();
		      if( parentId != p->pdgId()  && !(parentId<=100 && parentId>=80) ) 
			{
			  edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Id_parent = "<<parentId;
			  return parentId;
			}
		    }
		  motherMotherMotherMother = 0;
		}
	      motherMotherMother = 0; 
	    }// valid motherMother
	  motherMother = 0;
	}//valid mother 
      mother = 0;
    }//loop over tracking particles

  edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Done with getSimParentId!\t"<<parentId;
  return parentId;
}


// //_______________________________________________________________________
void McMatchTrackAnalyzer::dummVectorEntry(vector<float>& result, int nEntries)
{
   //   // add to vector nentries of '-99' 
  for(int i = 1; i<= nEntries; i++)
    result.push_back(-99);
}


//____________________________________________________________________________
void McMatchTrackAnalyzer::beginJob()
{
 // method called once each job just before starting event loop
  edm::LogInfo("MCMatchAnalyzer")<<"Begin job  ...";

  // Bookkeepping
  edm::Service<TFileService>   fs;

 //from reco2sim maps
  pnRecoSimTrkType1   = fs->make<TNtuple>("pnRecoSimTrkType1","pnRecoSimTrkType1",//50
					  "charge:chi2:chi2ndof:dxy:dxyerr:dz:dzerr:nvalidhits:eta:phi:pt:pterr:algo:" //12  reco track
					  "nhits:nvalidhits2:" //reco hits -- 22
					  "nvalidtrkerhits:nvalidpixelhits:nvalidstriphits:"
					  "nvalidmuonhits:nvaliddthits:nvalidcschits:nvalidrpchits:"
					  "nlosthits:"
					  "nlosttrkerhits:nlostpixelhits:nloststriphits:"
					  "nlostmuonhits:nlostdthits:nlostcschits:nlostrpchits:"
					  "nbadhits:"
					  "nbadmuonhits:nbaddthits:nbadcschits:nbadrpchits:"
					  "nmatch:frachitmatch:" //2
					  "chargesim:etasim:idsim:idparentsim:phisim:ptsim:"//6 matched sim sta
					  "nhitssim:ntrkerhitssim:npixelhitssim:nsilhitssim:"  // sim track 8
					  "nmuonhitssim:ndthitssim:ncschitssim:nrpchitssim:"
					  "cbin:bx:by:bz:vx:vy:vz"
					  );//matched sim track

  pnRecoSimTrkType2   = fs->make<TNtuple>("pnRecoSimTrkType2","pnRecoSimTrkType2",
					  "charge:chi2:chi2ndof:dxy:dxyerr:dz:dzerr:nvalidhits:eta:phi:pt:pterr:algo:" // reco track
					  "nhits:nvalidhits2:" //reco hits -- 22
					  "nvalidtrkerhits:nvalidpixelhits:nvalidstriphits:"
					  "nvalidmuonhits:nvaliddthits:nvalidcschits:nvalidrpchits:"
					  "nlosthits:"
					  "nlosttrkerhits:nlostpixelhits:nloststriphits:"
					  "nlostmuonhits:nlostdthits:nlostcschits:nlostrpchits:"
					  "nbadhits:"
					  "nbadmuonhits:nbaddthits:nbadcschits:nbadrpchits:"
					  "nmatch:frachitmatch:"
					  "chargesim:etasim:idsim:idparentsim:phisim:ptsim:"//matched sim sta
					  "nhitssim:ntrkerhitssim:npixelhitssim:nsilhitssim:"  // sim track 8
					  "nmuonhitssim:ndthitssim:ncschitssim:nrpchitssim:"
					  "cbin:bx:by:bz:vx:vy:vz"
					  );//matched sim track

  pnRecoSimPairType1 = fs->make<TNtuple>("pnRecoSimPairType1","pnRecoSimPairType1",
					 "y:minv:phi:pt:" // pair global  --4
					 "charge1:chi21:chi2ndof1:dxy1:dxyerr1:dz1:dzerr1:nvalidhits1:eta1:phi1:pt1:pt1err:algo1:"//kid1 --12
					 "nmatch1:frachitmatch1:"//matched sim kid 1  --9
					 "chargesim1:etasim1:idsim1:idparentsim1:phisim1:ptsim1:"
					 "charge2:chi22:chi2ndof2:dxy2:dxyerr2:dz2:dzerr2:nvalidhits2:eta2:phi2:pt2:pt2err:algo2:"//kid2 --12
					 "nmatch2:frachitmatch2:"// matched sim kid2  --9
					 "chargesim2:etasim2:idsim2:idparentsim2:phisim2:ptsim2:"
					 "cbin:bx:by:bz:vx:vy:vz"
					 );	
  
  pnRecoSimPairType2 = fs->make<TNtuple>("pnRecoSimPairType2","pnRecoSimPairType2",
					 "y:minv:phi:pt:" // pair global  --4
					 "charge1:chi21:chi2ndof1:dxy1:dxyerr1:dz1:dzerr1:nvalidhits1:eta1:phi1:pt1:pt1err:algo1:"//kid1 --12
					 "nmatch1:frachitmatch1:"//matched sim kid 1  --9
					 "chargesim1:etasim1:idsim1:idparentsim1:phisim1:ptsim1:"
					 "charge2:chi22:chi2ndof2:dxy2:dxyerr2:dz2:dzerr2:nvalidhits2:eta2:phi2:pt2:pt2err:algo2:"//kid2 --12
					 "nmatch2:frachitmatch2:"// matched sim kid2  --9
					 "chargesim2:etasim2:idsim2:idparentsim2:phisim2:ptsim2:"
					 "cbin:bx:by:bz:vx:vy:vz"
					 );


  // from sim2reco maps
  pnSimRecoTrkType1  = fs->make<TNtuple>("pnSimRecoTrkType1","pnSimRecoTrkType1",
					 "charge:dca:vtxz:vtxy:eta:id:idparent:phi:pt:status:"  //sim track --10
					 "nhits:ntrkerhits:npixelhits:nsilhits:"  // sim track 8
					 "nmuonhits:ndthits:ncschits:nrpchits:"
					 "nmatch:frachitmatch:" //# 2 of reco tracks matched to one sim track
					 "chargereco:chi2reco:chi2ndofreco:dxyreco:dxyerrreco:dzreco:dzerrreco:nvalidhitsreco:"
					 "etareco:phireco:ptreco:ptrecoerr:algoreco:"// 12
					 "nhitsreco:nvalidhitsreco2:" //reco hits -- 22
					 "nvalidtrkerhitsreco:nvalidpixelhitsreco:nvalidstriphitsreco:"
					 "nvalidmuonhitsreco:nvaliddthitsreco:nvalidcschitsreco:nvalidrpchitsreco:"
					 "nlosthitsreco:"
					 "nlosttrkerhitsreco:nlostpixelhitsreco:nloststriphitsreco:"
					 "nlostmuonhitsreco:nlostdthitsreco:nlostcschitsreco:nlostrpchitsreco:"
					 "nbadhitsreco:"
					 "nbadmuonhitsreco:nbaddthitsreco:nbadcschitsreco:nbadrpchitsreco:"
					 "cbin:bx:by:bz:vx:vy:vz"); // matched reco track
  pnSimRecoTrkType2  = fs->make<TNtuple>("pnSimRecoTrkType2","pnSimRecoTrkType2",
					 "charge:dca:vtxz:vtxy:eta:id:idparent:phi:pt:status:"  //sim track --12
					 "nhits:ntrkerhits:npixelhits:nsilhits:"  // sim track 8
					 "nmuonhits:ndthits:ncschits:nrpchits:"
					 "nmatch:frachitmatch:" //# of reco tracks matched to one sim track
					 "chargereco:chi2reco:chi2ndofreco:dxyreco:dxyerrreco:dzreco:dzerrreco:nvalidhitsreco:"
					 "etareco:phireco:ptreco:ptrecoerr:algoreco:"
					 "nhitsreco:nvalidhitsreco2:" //reco hits -- 22
					 "nvalidtrkerhitsreco:nvalidpixelhitsreco:nvalidstriphitsreco:"
					 "nvalidmuonhitsreco:nvaliddthitsreco:nvalidcschitsreco:nvalidrpchitsreco:"
					 "nlosthitsreco:"
					 "nlosttrkerhitsreco:nlostpixelhitsreco:nloststriphitsreco:"
					 "nlostmuonhitsreco:nlostdthitsreco:nlostcschitsreco:nlostrpchitsreco:"
					 "nbadhitsreco:"
					 "nbadmuonhitsreco:nbaddthitsreco:nbadcschitsreco:nbadrpchitsreco:"
					 "cbin:bx:by:bz:vx:vy:vz"); // matched reco track
  
  pnSimRecoPairType1 = fs->make<TNtuple>("pnSimRecoPairType1","pnSimRecoPairType1",
					 "y:minv:phi:pt:" // sim Z0 --4 
					 "charge1:dca1:vtxz1:vtxy1:eta1:id1:idparent1:phi1:pt1:status1:"  //sim mu1 --10
					 "nhits1:ntrkerhits1:npixelhits1:nsilhits1:"  // sim track 8
					 "nmuonhits1:ndthits1:ncschits1:nrpchits1:"
					 "nmatch1:frachitmatch1:" // reco muon1 -- 14
					 "chargereco1:chi2reco1:chi2ndofreco1:dxyreco1:dxyerrreco1:dzreco1:dzerrreco1:nvalidhitsreco1:etareco1:phireco1:ptreco1:ptrecoerr1:algoreco1:"
					 "nhitsreco1:nvalidhitsreco12:" //reco hits -- 22
					 "nvalidtrkerhitsreco1:nvalidpixelhitsreco1:nvalidstriphitsreco1:"
					 "nvalidmuonhitsreco1:nvaliddthitsreco1:nvalidcschitsreco1:nvalidrpchitsreco1:"
					 "nlosthitsreco1:"
					 "nlosttrkerhitsreco1:nlostpixelhitsreco1:nloststriphitsreco1:"
					 "nlostmuonhitsreco1:nlostdthitsreco1:nlostcschitsreco1:nlostrpchitsreco1:"
					 "nbadhitsreco1:"
					 "nbadmuonhitsreco1:nbaddthitsreco1:nbadcschitsreco1:nbadrpchitsreco1:"
					 "charge2:dca2:vtxz2:vtxy2:eta2:id2:idparent2:phi2:pt2:status2:"  //sim mu2 --10
					 "nhits2:ntrkerhits2:npixelhits2:nsilhits2:"  // sim track 8
					 "nmuonhits2:ndthits2:ncschits2:nrpchits2:"
					 "nmatch2:frachitmatch2:" // reco muon2--14
					 "chargereco2:chi2reco2:chi2ndofreco2:dxyreco2:dxyerrreco2:dzreco2:dzerrreco2:nvalidhitsreco2:etareco2:phireco2:ptreco2:ptrecoerr2:algoreco2:"
					 "nhitsreco2:nvalidhitsreco22:" //reco hits -- 22
					 "nvalidtrkerhitsreco2:nvalidpixelhitsreco2:nvalidstriphitsreco2:"
					 "nvalidmuonhitsreco2:nvaliddthitsreco2:nvalidcschitsreco2:nvalidrpchitsreco2:"
					 "nlosthitsreco2:"
					 "nlosttrkerhitsreco2:nlostpixelhitsreco2:nloststriphitsreco2:"
					 "nlostmuonhitsreco2:nlostdthitsreco2:nlostcschitsreco2:nlostrpchitsreco2:"
					 "nbadhitsreco2:"
					 "nbadmuonhitsreco2:nbaddthitsreco2:nbadcschitsreco2:nbadrpchitsreco2:"
					 "yreco:minvreco:phireco:ptreco:"// reco  Z0 
					 "cbin:bx:by:bz:vx:vy:vz"
					 ); // reco Z0 from trk
  
  pnSimRecoPairType2 = fs->make<TNtuple>("pnSimRecoPairType2","pnSimRecoPairType2",
					 "y:minv:phi:pt:" // sim Z0 --4 
					 "charge1:dca1:vtxz1:vtxy1:eta1:id1:idparent1:phi1:pt1:status1:"  //sim mu1 --10
					 "nhits1:ntrkerhits1:npixelhits1:nsilhits1:"  // sim track 8
					 "nmuonhits1:ndthits1:ncschits1:nrpchits1:"
					 "nmatch1:frachitmatch1:" // reco muon1 -- 14
					 "chargereco1:chi2reco1:chi2ndofreco1:dxyreco1:dxyerrreco1:dzreco1:dzerrreco1:nvalidhitsreco1:etareco1:phireco1:ptreco1:ptrecoerr1:algoreco1:"
					 "nhitsreco1:nvalidhitsreco12:" //reco hits -- 22
					 "nvalidtrkerhitsreco1:nvalidpixelhitsreco1:nvalidstriphitsreco1:"
					 "nvalidmuonhitsreco1:nvaliddthitsreco1:nvalidcschitsreco1:nvalidrpchitsreco1:"
					 "nlosthitsreco1:"
					 "nlosttrkerhitsreco1:nlostpixelhitsreco1:nloststriphitsreco1:"
					 "nlostmuonhitsreco1:nlostdthitsreco1:nlostcschitsreco1:nlostrpchitsreco1:"
					 "nbadhitsreco1:"
					 "nbadmuonhitsreco1:nbaddthitsreco1:nbadcschitsreco1:nbadrpchitsreco1:"
					 "charge2:dca2:vtxz2:vtxy2:eta2:id2:idparent2:phi2:pt2:status2:"  //sim mu2 --10
					 "nhits2:ntrkerhits2:npixelhits2:nsilhits2:"  // sim track 8
					 "nmuonhits2:ndthits2:ncschits2:nrpchits2:"
					 "nmatch2:frachitmatch2:" // reco muon2--14
					 "chargereco2:chi2reco2:chi2ndofreco2:dxyreco2:dxyerrreco2:dzreco2:dzerrreco2:nvalidhitsreco2:etareco2:phireco2:ptreco2:ptrecoerr2:algoreco2:"
					 "nhitsreco2:nvalidhitsreco22:" //reco hits -- 22
					 "nvalidtrkerhitsreco2:nvalidpixelhitsreco2:nvalidstriphitsreco2:"
					 "nvalidmuonhitsreco2:nvaliddthitsreco2:nvalidcschitsreco2:nvalidrpchitsreco2:"
					 "nlosthitsreco2:"
					 "nlosttrkerhitsreco2:nlostpixelhitsreco2:nloststriphitsreco2:"
					 "nlostmuonhitsreco2:nlostdthitsreco2:nlostcschitsreco2:nlostrpchitsreco2:"
					 "nbadhitsreco2:"
					 "nbadmuonhitsreco2:nbaddthitsreco2:nbadcschitsreco2:nbadrpchitsreco2:"
					 "yreco:minvreco:phireco:ptreco:"// reco  Z0 
					 "cbin:bx:by:bz:vx:vy:vz"//event info
					 ); // reco Z0 from trk
   
  pnEventInfo      = fs->make<TNtuple>("pnEventInfo","pnEventInfo","proc:procsgn:nsimtrk:ntrk1:ntrk2:nvtx:ncoll:nhard:npart:b:cbin:bx:by:bz:vx:vy:vz");

  edm::LogInfo("MCMatchAnalyzer")<<"End beginning job ...";

}


//_________________________________________________________________________
void McMatchTrackAnalyzer::endJob()
{
  //method called once each job just after ending the event loop 
  edm::LogInfo("MCMatchAnalyzer")<<"End job ...";

}


//________________________________________________________________________
std::vector<PSimHit> McMatchTrackAnalyzer::getPSimHits(const TrackingParticleRef& st, DetId::Detector detector)
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
std::vector<PSimHit> McMatchTrackAnalyzer::getPSimHits(const TrackingParticleRef& st)
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

DEFINE_FWK_MODULE(McMatchTrackAnalyzer);

//========================================

//=======================================
// int McMatchTrackAnalyzer::getSimParentId(const TrackingParticleRef& match)
// {
//   // return the particle ID of associated GEN track (trackingparticle = gen + sim(geant) )
//   // it is a final, stable particle (status=1): after final state radiaiton

//   // same particle before final state radiation (status=3); from this one get to the real parent;
//   // this is the 'parent' of the final state particle, p; same pdg_id
  
//   int parentId = -99; 
  
//   if( match.isNull() ) return parentId;
   
//   edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Getting the parent Id for the sim particle with status..."<< match->status();
//   TrackingParticle::genp_iterator b;
//   TrackingParticle::genp_iterator in  = match->genParticle_begin();
//   TrackingParticle::genp_iterator fin = match->genParticle_end();
//   for (b = in; b != fin; ++b)
//     {
//       const HepMC::GenParticle *p = b->get();
//       if( p==NULL ) 
// 	{
// 	  edm::LogInfo("MCMatchAnalyzer::getSimParentId()")<<"No gen particle associated with simTrack with status " << match->status();
// 	  continue;
// 	}
  
//       edm::LogInfo("MCMatchAnalyzer::getSimParentId()")<<"Gen particle is : " << (*b)->pdg_id();
//       const HepMC::GenParticle *mother = p->production_vertex() && 
// 	p->production_vertex()->particles_in_const_begin() != p->production_vertex()->particles_in_const_end() ?
// 	*(p->production_vertex()->particles_in_const_begin()) : 0;
    
//       if( mother!=0 && !(std::isnan(mother->momentum().perp())) && !std::isnan(abs(mother->pdg_id())) )
// 	{
	  
// 	  if( abs(mother->pdg_id())<10 || (mother->pdg_id()<=100 && mother->pdg_id()>=80) || mother->pdg_id()==21)
// 	    { 
// 	      edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Id_parent = 0; comes from PV directly ";
// 	      return mother->pdg_id(); // from PV (parton or cluster)
// 	    }
// 	  else
// 	    {
// 	      parentId = mother->pdg_id();
// 	      if( parentId != p->pdg_id()) 
// 		{
// 		  edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Id_parent = "<<parentId;
// 		  return parentId;
// 		}
// 	      edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<" 1 parentId = "<<parentId<<"\t parent_status= "<<mother->status();
	  
// 	      // real parent after final state radiation
// 	      const HepMC::GenParticle *motherMother = mother->production_vertex() &&
// 		mother->production_vertex()->particles_in_const_begin() != mother->production_vertex()->particles_in_const_end() ?
// 		*(mother->production_vertex()->particles_in_const_begin()) : 0 ;
	      
// 	      if( motherMother!=0  
// 		  && !(std::isnan(motherMother->momentum().perp()) ) && !(std::isnan(motherMother->pdg_id()) ) )
// 		{ 
// 		  if(abs(motherMother->pdg_id())<10 || 
// 		     (motherMother->pdg_id()<=100 && motherMother->pdg_id()>=80) || 
// 		     motherMother->pdg_id()==21)
// 		    {
// 		      edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Id_parent = 0; comes from PV indirectly ";
// 		      return motherMother->pdg_id();
// 		    }
// 		  else
// 		    {
// 		      parentId = motherMother->pdg_id();
// 		      edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<" 2 parentId = "<<parentId<<"\t parent_status= "<<motherMother->status();
// 		    }
// 		  motherMother = 0;
// 		}// valid motherMother
// 	    }//else: mother not from PV
// 	}//valid mother 
//      mother = 0;
//     }//loop over tracking particles

//   edm::LogInfo("McMatchTrackAnalyzer::getSimParentId")<<"Done with getSimParentId!\t"<<parentId;
//   return parentId;
// }
