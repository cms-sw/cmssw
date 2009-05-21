// -*- C++ -*-
//
// Package:    V0Validator
// Class:      V0Validator
// 
/**\class V0Validator V0Validator.cc Validation/RecoVertex/src/V0Validator.cc

 Description: Creates validation histograms for RecoVertex/V0Producer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Wed Feb 18 17:21:04 MST 2009
// $Id: V0Validator.cc,v 1.2 2009/05/19 00:47:59 drell Exp $
//
//


#include "Validation/RecoVertex/interface/V0Validator.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

typedef std::vector<TrackingVertex> TrackingVertexCollection;
typedef edm::Ref<TrackingVertexCollection> TrackingVertexRef;
typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
typedef edm::RefVector<edm::HepMCProduct, HepMC::GenParticle > GenParticleRefVector;

const double piMass = 0.13957018;
const double piMassSquared = piMass*piMass;
const double protonMass = 0.93827203;
const double protonMassSquared = protonMass*protonMass;



V0Validator::V0Validator(const edm::ParameterSet& iConfig) : 
  theDQMRootFileName(iConfig.getParameter<std::string>("DQMRootFileName")) {
  genLam = genK0s = realLamFoundEff = realK0sFoundEff = lamCandFound = 
    k0sCandFound = noTPforK0sCand = noTPforLamCand = realK0sFound = realLamFound = 0;
}


V0Validator::~V0Validator() {

}

void V0Validator::beginJob(const edm::EventSetup&) {
  theDQMstore = edm::Service<DQMStore>().operator->();
  //std::cout << "In beginJob() at line 1" << std::endl;
  //edm::Service<TFileService> fs;

  ksEffVsRHist = new TH1F("K0sEffVsR", 
			  "K^{0}_{s} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEtaHist = new TH1F("K0sEffVsEta",
			    "K^{0}_{s} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPtHist = new TH1F("K0sEffVsPt",
			   "K^{0}_{s} Efficiency vs p_{T}", 70, 0., 20.);;
  ksFakeVsRHist = new TH1F("K0sFakeVsR",
			   "K^{0}_{s} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEtaHist = new TH1F("K0sFakeVsEta",
			     "K^{0}_{s} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPtHist = new TH1F("K0sFakeVsPt",
			    "K^{0}_{s} Fake Rate vs p_{T}", 70, 0., 20.);

  ksTkEffVsRHist = new TH1F("K0sTkEffVsR", 
			  "K^{0}_{s} Tracking Efficiency vs #rho", 40, 0., 40.);
  ksTkEffVsEtaHist = new TH1F("K0sTkEffVsEta",
			    "K^{0}_{s} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  ksTkEffVsPtHist = new TH1F("K0sTkEffVsPt",
			   "K^{0}_{s} Tracking Efficiency vs p_{T}", 70, 0., 20.);;
  ksTkFakeVsRHist = new TH1F("K0sTkFakeVsR",
			   "K^{0}_{s} Tracking Fake Rate vs #rho", 40, 0., 40.);
  ksTkFakeVsEtaHist = new TH1F("K0sTkFakeVsEta",
			     "K^{0}_{s} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  ksTkFakeVsPtHist = new TH1F("K0sTkFakeVsPt",
			    "K^{0}_{s} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  ksEffVsRHist_denom = new TH1F("K0sEffVsR_denom", 
			  "K^{0}_{s} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEtaHist_denom = new TH1F("K0sEffVsEta_denom",
			    "K^{0}_{s} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPtHist_denom = new TH1F("K0sEffVsPt_denom",
			   "K^{0}_{s} Efficiency vs p_{T}", 70, 0., 20.);;
  ksFakeVsRHist_denom = new TH1F("K0sFakeVsR_denom",
			   "K^{0}_{s} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEtaHist_denom = new TH1F("K0sFakeVsEta_denom",
			     "K^{0}_{s} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPtHist_denom = new TH1F("K0sFakeVsPt_denom",
			    "K^{0}_{s} Fake Rate vs p_{T}", 70, 0., 20.);

  lamEffVsRHist = new TH1F("LamEffVsR",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEtaHist = new TH1F("LamEffVsEta",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPtHist = new TH1F("LamEffVsPt",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);
  lamFakeVsRHist = new TH1F("LamFakeVsR",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEtaHist = new TH1F("LamFakeVsEta",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPtHist = new TH1F("LamFakeVsPt",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);

  lamTkEffVsRHist = new TH1F("LamTkEffVsR",
			   "#Lambda^{0} TrackingEfficiency vs #rho", 40, 0., 40.);
  lamTkEffVsEtaHist = new TH1F("LamTkEffVsEta",
			     "#Lambda^{0} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  lamTkEffVsPtHist = new TH1F("LamTkEffVsPt",
			    "#Lambda^{0} Tracking Efficiency vs p_{T}", 70, 0., 20.);
  lamTkFakeVsRHist = new TH1F("LamTkFakeVsR",
			    "#Lambda^{0} Tracking Fake Rate vs #rho", 40, 0., 40.);
  lamTkFakeVsEtaHist = new TH1F("LamTkFakeVsEta",
			      "#Lambda^{0} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  lamTkFakeVsPtHist = new TH1F("LamTkFakeVsPt",
			     "#Lambda^{0} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  lamEffVsRHist_denom = new TH1F("LamEffVsR_denom",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEtaHist_denom = new TH1F("LamEffVsEta_denom",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPtHist_denom = new TH1F("LamEffVsPt_denom",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);
  lamFakeVsRHist_denom = new TH1F("LamFakeVsR_denom",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEtaHist_denom = new TH1F("LamFakeVsEta_denom",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPtHist_denom = new TH1F("LamFakeVsPt_denom",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);

  nKsHist = new TH1F("nK0s",
		     "Number of K^{0}_{s} found per event", 60, 0., 60.);
  nLamHist = new TH1F("nLam",
		      "Number of #Lambda^{0} found per event", 60, 0., 60.);

  ksXResolutionHist = new TH1F("ksXResolution",
			       "Resolution of V0 decay vertex X coordinate", 50, 0., 50.);
  ksYResolutionHist = new TH1F("ksYResolution",
			       "Resolution of V0 decay vertex Y coordinate", 50, 0., 50.);
  ksZResolutionHist = new TH1F("ksZResolution",
			       "Resolution of V0 decay vertex Z coordinate", 50, 0., 50.);
  lamXResolutionHist = new TH1F("lamXResolution",
				"Resolution of V0 decay vertex X coordinate", 50, 0., 50.);
  lamYResolutionHist = new TH1F("lamYResolution",
				"Resolution of V0 decay vertex Y coordinate", 50, 0., 50.);
  lamZResolutionHist = new TH1F("lamZResolution",
				"Resolution of V0 decay vertex Z coordinate", 50, 0., 50.);
  ksAbsoluteDistResolutionHist = new TH1F("ksRResolution",
					  "Resolution of absolute distance from primary vertex to V0 vertex",
					  100, 0., 50.);
  lamAbsoluteDistResolutionHist = new TH1F("lamRResolution",
					   "Resolution of absolute distance from primary vertex to V0 vertex",
					   100, 0., 50.);



  //std::cout << "Histograms booked" << std::endl;

  ksEffVsRHist->Sumw2();
  ksEffVsEtaHist->Sumw2();
  ksEffVsPtHist->Sumw2();
  ksTkEffVsRHist->Sumw2();
  ksTkEffVsEtaHist->Sumw2();
  ksTkEffVsPtHist->Sumw2();
  ksFakeVsRHist->Sumw2();
  ksFakeVsEtaHist->Sumw2();
  ksFakeVsPtHist->Sumw2();
  ksTkFakeVsRHist->Sumw2();
  ksTkFakeVsEtaHist->Sumw2();
  ksTkFakeVsPtHist->Sumw2();

  lamEffVsRHist->Sumw2();
  lamEffVsEtaHist->Sumw2();
  lamEffVsPtHist->Sumw2();
  lamTkEffVsRHist->Sumw2();
  lamTkEffVsEtaHist->Sumw2();
  lamTkEffVsPtHist->Sumw2();
  lamFakeVsRHist->Sumw2();
  lamFakeVsEtaHist->Sumw2();
  lamFakeVsPtHist->Sumw2();
  lamTkFakeVsRHist->Sumw2();
  lamTkFakeVsEtaHist->Sumw2();
  lamTkFakeVsPtHist->Sumw2();

}

void V0Validator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using std::cout;
  using std::endl;
  using namespace edm;
  using namespace std;

  //cout << "In analyze()" << endl;
  // Get event setup info, B-field and tracker geometry
  ESHandle<MagneticField> bFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

  // Make matching collections
  reco::RecoToSimCollection recSimColl;
  reco::SimToRecoCollection simRecColl;
   
  Handle<reco::RecoToSimCollection > recotosimCollectionH;
  iEvent.getByLabel("trackingParticleRecoTrackAsssociation", recotosimCollectionH);
  recSimColl= *( recotosimCollectionH.product() ); 
  
  Handle<reco::SimToRecoCollection> simtorecoCollectionH;
  iEvent.getByLabel("trackingParticleRecoTrackAsssociation", simtorecoCollectionH);
  simRecColl= *( simtorecoCollectionH.product() );

  edm::Handle<TrackingParticleCollection>  TPCollectionEff ;
  iEvent.getByLabel("mergedtruth", "MergedTrackTruth", TPCollectionEff);
  const TrackingParticleCollection tPCeff = *( TPCollectionEff.product() );

  edm::ESHandle<TrackAssociatorBase> associatorByHits;
  iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits", associatorByHits);

  VertexAssociatorBase* associatorByTracks;

  edm::ESHandle<VertexAssociatorBase> theTracksAssociator;
  iSetup.get<VertexAssociatorRecord>().get("VertexAssociatorByTracks",theTracksAssociator);
  associatorByTracks = (VertexAssociatorBase *) theTracksAssociator.product();

  // Get tracks
  Handle< View<reco::Track> > trackCollectionH;
  iEvent.getByLabel("generalTracks", trackCollectionH);

  Handle<SimTrackContainer> simTrackCollection;
  iEvent.getByLabel("g4SimHits", simTrackCollection);
  const SimTrackContainer simTC = *(simTrackCollection.product());

  Handle<SimVertexContainer> simVertexCollection;
  iEvent.getByLabel("g4SimHits", simVertexCollection);
  const SimVertexContainer simVC = *(simVertexCollection.product());

  //Get tracking particles
  //  -->tracks
  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  iEvent.getByLabel("mergedtruth", "MergedTrackTruth", TPCollectionH);
  const View<reco::Track>  tC = *( trackCollectionH.product() );

//  edm::Handle<TrackingVertexCollection>  TVCollectionH ;
//  iEvent.getByLabel("trackingParticles","VertexTruth",TVCollectionH);
//  const TrackingVertexCollection tVC   = *(TVCollectionH.product());

  // Select the primary vertex, create a new reco::Vertex to hold it
  edm::Handle< std::vector<reco::Vertex> > primaryVtxCollectionH;
  iEvent.getByLabel("offlinePrimaryVertices", primaryVtxCollectionH);
  const reco::VertexCollection primaryVertexCollection   = *(primaryVtxCollectionH.product());

  reco::Vertex* thePrimary = 0;
  std::vector<reco::Vertex>::const_iterator iVtxPH = primaryVtxCollectionH->begin();
  for(std::vector<reco::Vertex>::const_iterator iVtx = primaryVtxCollectionH->begin();
      iVtx < primaryVtxCollectionH->end();
      iVtx++) {
    if(primaryVtxCollectionH->size() > 1) {
      if(iVtx->tracksSize() > iVtxPH->tracksSize()) {
	iVtxPH = iVtx;
      }
    }
    else iVtxPH = iVtx;
  }
  thePrimary = new reco::Vertex(*iVtxPH);
 

  reco::RecoToSimCollection r2s = associatorByHits->associateRecoToSim(trackCollectionH,TPCollectionH,&iEvent );
  reco::SimToRecoCollection s2r = associatorByHits->associateSimToReco(trackCollectionH,TPCollectionH,&iEvent );

//  reco::VertexRecoToSimCollection vr2s = associatorByTracks->associateRecoToSim(primaryVtxCollectionH, TVCollectionH, iEvent, r2s);
//  reco::VertexSimToRecoCollection vs2r = associatorByTracks->associateSimToReco(primaryVtxCollectionH, TVCollectionH, iEvent, s2r);

  //get the V0s;   
  edm::Handle<reco::VertexCompositeCandidateCollection> k0sCollection;
  edm::Handle<reco::VertexCompositeCandidateCollection> lambdaCollection;
  iEvent.getByLabel("generalV0Candidates", "Kshort", k0sCollection);
  iEvent.getByLabel("generalV0Candidates", "Lambda", lambdaCollection);

  //make vector of pair of trackingParticles to hold good V0 candidates
  std::vector< pair<TrackingParticleRef, TrackingParticleRef> > trueK0s;
  std::vector< pair<TrackingParticleRef, TrackingParticleRef> > trueLams;

  ////////////////////////////
  // Do vertex calculations //
  ////////////////////////////
/*
  if( k0sCollection->size() > 0 ) {
    for(reco::VertexCompositeCandidateCollection::const_iterator iK0s = k0sCollection->begin();
	iK0s != k0sCollection->end();
	iK0s++) {
      // Still can't actually associate the V0 vertex with a TrackingVertexCollection.
      //  Is this a problem?  You bet.
      reco::VertexCompositeCandidate::CovarianceMatrix aErr;
      iK0s->fillVertexCovariance(aErr);
      reco::Vertex tVtx(iK0s->vertex(), aErr);
      reco::VertexCollection *tVtxColl = 0;
      tVtxColl->push_back(tVtx);
      reco::VertexRef aVtx(tVtxColl, 0);
      //if(vr2s.find(iK0s->vertex()) != vr2s.end()) {
      if(vr2s.find(aVtx) != vr2s.end()) {
	cout << "Found it in the collection." << endl;
      	std::vector< std::pair<TrackingVertexRef, double> > vVR 
	  = (std::vector< std::pair<TrackingVertexRef, double> >) vr2s[aVtx];
      }
    }
  }
*/
  //////////////////////////////
  // Do fake rate calculation //
  //////////////////////////////

  //cout << "Starting K0s fake rate calculation" << endl;
  // Kshorts
  double numK0sFound = 0.;
  if ( k0sCollection->size() > 0 ) {

    vector<reco::TrackRef> theDaughterTracks;
    for( reco::VertexCompositeCandidateCollection::const_iterator iK0s = k0sCollection->begin();
	 iK0s != k0sCollection->end();
	 iK0s++) {
      // Fill values to be histogrammed
      K0sCandpT = (sqrt( iK0s->momentum().perp2() ));
      K0sCandEta = iK0s->momentum().eta();
      K0sCandR = (sqrt( iK0s->vertex().perp2() ));
      K0sCandStatus = 0;

      theDaughterTracks.push_back( (*(dynamic_cast<const reco::RecoChargedCandidate *> (iK0s->daughter(0)) )).track() );
      theDaughterTracks.push_back( (*(dynamic_cast<const reco::RecoChargedCandidate *> (iK0s->daughter(1)) )).track() );
       
      for (int itrack = 0; itrack < 2; itrack++) {
	K0sPiCandStatus[itrack] = 0;
      }

      std::vector< std::pair<TrackingParticleRef, double> > tp;
      TrackingParticleRef tpref;
      TrackingParticleRef firstDauTP;
      TrackingVertexRef k0sVtx;

      // Loop through K0s candidate daugher tracks
      for(View<reco::Track>::size_type i=0; i<theDaughterTracks.size(); ++i){
	// Found track from theDaughterTracks
	RefToBase<reco::Track> track( theDaughterTracks.at(i) );

	if(recSimColl.find(track) != recSimColl.end()) {
	  tp = recSimColl[track];
	  if (tp.size() != 0) {
	    K0sPiCandStatus[i] = 1;
	    tpref = tp.begin()->first;

	    if( simRecColl.find(tpref) == simRecColl.end() ) {
	      K0sPiCandStatus[i] = 3;
	    }
	    TrackingVertexRef parentVertex = tpref->parentVertex();
	     
	    if( parentVertex.isNonnull() ) {
	      if( k0sVtx.isNonnull() ) {
		if( k0sVtx->position() == parentVertex->position() ) {
		  if( parentVertex->nDaughterTracks() == 2 ) {
		    if( parentVertex->nSourceTracks() == 0 ) {
		      // No source tracks found for K0s vertex; shouldn't happen, but does for evtGen events
		      K0sCandStatus = 6;
		    }
		    
		    for( TrackingVertex::tp_iterator iTP = parentVertex->sourceTracks_begin();
			 iTP != parentVertex->sourceTracks_end(); iTP++) {
		      if( (*iTP)->pdgId() == 310 ) {
			K0sCandStatus = 1;
			realK0sFound++;
			numK0sFound += 1.;
			std::pair<TrackingParticleRef, TrackingParticleRef> pair(firstDauTP, tpref);
			// Pushing back a good V0
			trueK0s.push_back(pair);
		      }
		      else {
			K0sCandStatus = 2;
		      }
		    }
		  }
		  else {
		    // Found a bad match because the mother has too many daughters
		    K0sCandStatus = 3;
		  }
		}
		else {
		  // Found a bad match because the parent vertices from the two tracks are different
		  K0sCandStatus = 4;
		}
	      }
	      else {
		// if k0sVtx is null, fill it with parentVertex to compare to the parentVertex from the second track
		k0sVtx = parentVertex;
		firstDauTP = tpref;
	      }
	    }//parent vertex is null
	  }//tp size zero
	}
	else {
	  K0sPiCandStatus[i] = 2;
	  noTPforK0sCand++;
	  K0sCandStatus = 5;
	  theDaughterTracks.clear();
	}
      }
      theDaughterTracks.clear();
      // fill the fake rate histograms
      if( K0sCandStatus > 1 ) {
	ksFakeVsRHist->Fill(K0sCandR);
	ksFakeVsEtaHist->Fill(K0sCandEta);
	ksFakeVsPtHist->Fill(K0sCandpT);
      }
      if( K0sCandStatus == 5 ) {
	ksTkFakeVsRHist->Fill(K0sCandR);
	ksTkFakeVsEtaHist->Fill(K0sCandEta);
	ksTkFakeVsPtHist->Fill(K0sCandpT);
      }
      ksFakeVsRHist_denom->Fill(K0sCandR);
      ksFakeVsEtaHist_denom->Fill(K0sCandEta);
      ksFakeVsPtHist_denom->Fill(K0sCandpT);
    }
  }
  //double numK0sFound = (double) realK0sFound;
  nKsHist->Fill( (double) numK0sFound );
  numK0sFound = 0.;

  //cout << "Starting Lambda fake rate calculation" << endl;

  double numLamFound = 0.;
  // Lambdas
  if ( lambdaCollection->size() > 0 ) {
    
    vector<reco::TrackRef> theDaughterTracks;
    for( reco::VertexCompositeCandidateCollection::const_iterator iLam = lambdaCollection->begin();
	 iLam != lambdaCollection->end();
	 iLam++) {
      // Fill values to be histogrammed
      LamCandpT = (sqrt( iLam->momentum().perp2() ));
      LamCandEta = iLam->momentum().eta();
      LamCandR = (sqrt( iLam->vertex().perp2() ));
      LamCandStatus = 0;
      
      theDaughterTracks.push_back( (*(dynamic_cast<const reco::RecoChargedCandidate *> (iLam->daughter(0)) )).track() );
      theDaughterTracks.push_back( (*(dynamic_cast<const reco::RecoChargedCandidate *> (iLam->daughter(1)) )).track() );
      
      for (int itrack = 0; itrack < 2; itrack++) {
	LamPiCandStatus[itrack] = 0;
      }
      
      std::vector< std::pair<TrackingParticleRef, double> > tp;
      TrackingParticleRef tpref;
      TrackingParticleRef firstDauTP;
      TrackingVertexRef LamVtx;
      // Loop through Lambda candidate daughter tracks
      for(View<reco::Track>::size_type i=0; i<theDaughterTracks.size(); ++i){
	// Found track from theDaughterTracks
	RefToBase<reco::Track> track( theDaughterTracks.at(i) );
	
	if(recSimColl.find(track) != recSimColl.end()) {
	  tp = recSimColl[track];
	  if (tp.size() != 0) {
	    LamPiCandStatus[i] = 1;
	    tpref = tp.begin()->first;

	    if( simRecColl.find(tpref) == simRecColl.end() ) {
	      LamPiCandStatus[i] = 3;
	    }
	    TrackingVertexRef parentVertex = tpref->parentVertex();
	     
	    if( parentVertex.isNonnull() ) {
	      if( LamVtx.isNonnull() ) {
		if( LamVtx->position() == parentVertex->position() ) {
		  if( parentVertex->nDaughterTracks() == 2 ) {
		    if( parentVertex->nSourceTracks() == 0 ) {
		      // No source tracks found for K0s vertex; shouldn't happen, but does for evtGen events
		      LamCandStatus = 6;
		    }

		    for( TrackingVertex::tp_iterator iTP = parentVertex->sourceTracks_begin();
			 iTP != parentVertex->sourceTracks_end(); ++iTP) {
		      if( abs((*iTP)->pdgId()) == 3122 ) {
			LamCandStatus = 1;
			realLamFound++;
			numLamFound += 1.;
			std::pair<TrackingParticleRef, TrackingParticleRef> pair(firstDauTP, tpref);
			// Pushing back a good V0
			trueLams.push_back(pair);
		      }
		      else {
			LamCandStatus = 2;
		      }
		      //if(iTP != parentVertex->sourceTracks_end()) {
		      //cout << "Bogus check 1" << endl;
		      //}
		    }
		  }
		  else {
		    // Found a bad match because the mother has too many daughters
		    LamCandStatus = 3;
		  }
		}
		else {
		  // Found a bad match because the parent vertices from the two tracks are different
		  LamCandStatus = 4;
		}
	      }
	      else {
		// if lamVtx is null, fill it with parentVertex to compare to the parentVertex from the second track
		LamVtx = parentVertex;
		firstDauTP = tpref;
	      }
	    }//parent vertex is null
	  }//tp size zero
	}
	else {
	  LamPiCandStatus[i] = 2;
	  noTPforLamCand++;
	  LamCandStatus = 5;
	  theDaughterTracks.clear();
	}
      }
      theDaughterTracks.clear();
      // fill the fake rate histograms
      if( LamCandStatus > 1 ) {
	lamFakeVsRHist->Fill(LamCandR);
	lamFakeVsEtaHist->Fill(LamCandEta);
	lamFakeVsPtHist->Fill(LamCandpT);
      }
      if( K0sCandStatus == 5 ) {
	lamTkFakeVsRHist->Fill(LamCandR);
	lamTkFakeVsEtaHist->Fill(LamCandEta);
	lamTkFakeVsPtHist->Fill(LamCandpT);
      }
      lamFakeVsRHist_denom->Fill(LamCandR);
      lamFakeVsEtaHist_denom->Fill(LamCandEta);
      lamFakeVsPtHist_denom->Fill(LamCandpT);
    }
  }
  nLamHist->Fill( (double) numLamFound );
  numLamFound = 0.;


  ///////////////////////////////
  // Do efficiency calculation //
  ///////////////////////////////

  //cout << "Starting Lambda efficiency" << endl;
  // Lambdas

  for(TrackingParticleCollection::size_type i = 0; i < tPCeff.size(); i++) {
    TrackingParticleRef tpr1(TPCollectionEff, i);
    TrackingParticle* itp1 = const_cast<TrackingParticle*>(tpr1.get());
    if( (itp1->pdgId() == 211 || itp1->pdgId() == 2212)
	&& itp1->parentVertex().isNonnull()
	&& abs(itp1->momentum().eta()) < 2.4
	&& sqrt( itp1->momentum().perp2() ) > 0.9) {
      bool isLambda = false;
      if( itp1->pdgId() == 2212 ) isLambda = true;
      if( itp1->parentVertex()->nDaughterTracks() == 2 ) {

	TrackingVertexRef piCand1Vertex = itp1->parentVertex();
	for(TrackingVertex::tp_iterator iTP1 = piCand1Vertex->sourceTracks_begin();
	    iTP1 != piCand1Vertex->sourceTracks_end(); iTP1++) {
	  if( abs((*iTP1)->pdgId()) == 3122 ) {
	    //double motherpT = (*iTP1)->pt();
	    //	     ----->>>>>>Keep going here
	    for(TrackingParticleCollection::size_type j=0;
		j < tPCeff.size();
		j++) {
	      TrackingParticleRef tpr2(TPCollectionEff, j);
	      TrackingParticle* itp2 = const_cast<TrackingParticle*>(tpr2.get());
	      int particle2pdgId;
	      if (isLambda) particle2pdgId = -211;
	      else particle2pdgId = -2212;
	      if( itp2->pdgId() == particle2pdgId
		  && itp2->parentVertex().isNonnull()
		  && abs(itp2->momentum().eta()) < 2.4
		  && sqrt(itp2->momentum().perp2()) > 0.9) {
		if(itp2->parentVertex() == itp1->parentVertex()) {
		  // Found a good pair of Lambda daughters
		  TrackingVertexRef piCand2Vertex = itp2->parentVertex();
		  for (TrackingVertex::tp_iterator iTP2 = piCand2Vertex->sourceTracks_begin();
		       iTP2 != piCand2Vertex->sourceTracks_end(); 
		       ++iTP2) {
		    LamGenEta = LamGenpT = LamGenR = 0.;
		    LamGenStatus = 0;
		    for(int ifill = 0;
			ifill < 2;
			ifill++) {
		      // do nothing?
		    }
		    if( abs((*iTP2)->pdgId()) == 3122 ) {
		      // found generated Lambda
		      
		      LamGenpT = sqrt((*iTP2)->momentum().perp2());
		      LamGenEta = (*iTP2)->momentum().eta();
		      LamGenR = sqrt(itp2->vertex().perp2());
		      genLam++;
		      if(trueLams.size() > 0) {
			for(std::vector< pair<TrackingParticleRef, TrackingParticleRef> >::const_iterator iEffCheck = trueLams.begin();
			    iEffCheck != trueLams.end();
			    iEffCheck++) {
			  cout << "In LOOP" << endl;
			  if( itp1->parentVertex() == iEffCheck->first->parentVertex()
			      && itp2->parentVertex() == iEffCheck->second->parentVertex() ) {
			    realLamFoundEff++;
			    //V0Producer found the generated Lambda
			    LamGenStatus = 1;
			    break;
			  }
			  else {
			    //V0Producer didn't find the generated Lambda
			    LamGenStatus = 2;
			  }
			}
		      }
		      else {
			//No V0 cand found, so V0Producer didn't find the generated Lambda
			LamGenStatus = 2;
		      }
		      std::vector< std::pair<RefToBase<reco::Track>, double> > rt1;
		      std::vector< std::pair<RefToBase<reco::Track>, double> > rt2;
		      
		      if( simRecColl.find(tpr1) != simRecColl.end() ) {
			rt1 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) simRecColl[tpr1];
			if(rt1.size() != 0) {
			  LamPiEff[0] = 1; //Found the first daughter track
			  edm::RefToBase<reco::Track> t1 = rt1.begin()->first;
			}
		      }
		      else {
			LamPiEff[0] = 2;//First daughter not found
		      }
		      if( (simRecColl.find(tpr2) != simRecColl.end()) ) {
			rt2 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) simRecColl[tpr2];
			if(rt2.size() != 0) {
			  LamPiEff[1] = 1;//Found the second daughter track
			  edm::RefToBase<reco::Track> t2 = rt2.begin()->first;
			}
		      }
		      else {
			LamPiEff[1] = 2;//Second daughter not found
		      }
		      
		      if( LamGenStatus == 1
			  && (LamPiEff[0] == 2 || LamPiEff[1] == 2) ) {
			// Good Lambda found, but recoTrack->trackingParticle->recoTrack didn't work
			LamGenStatus = 4;
			realLamFoundEff--;
		      }
		      if( LamGenStatus == 2
			  && (LamPiEff[0] == 2 || LamPiEff[1] == 2) ) {
			// Lambda not found because we didn't find a daughter track
			LamGenStatus = 3;
		      }
		      cout << "LamGenStatus: " << LamGenStatus << ", LamPiEff[i]: " << LamPiEff[0] << ", " << LamPiEff[1] << endl;
		      // Fill histograms
		      if(LamGenR > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsRHist->Fill(LamGenR);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsRHist->Fill(LamGenR);
			}
			lamEffVsRHist_denom->Fill(LamGenR);
		      }
		      if(abs(LamGenEta) > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsEtaHist->Fill(LamGenEta);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsEtaHist->Fill(LamGenEta);
			}
			lamEffVsEtaHist_denom->Fill(LamGenEta);
		      }
		      if(LamGenpT > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsPtHist->Fill(LamGenpT);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsPtHist->Fill(LamGenpT);
			}
			lamEffVsPtHist_denom->Fill(LamGenpT);
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  //Kshorts

  //cout << "Starting Kshort efficiency" << endl;
  for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
    TrackingParticleRef tpr1(TPCollectionEff, i);
    TrackingParticle* itp1=const_cast<TrackingParticle*>(tpr1.get());
    // only count the efficiency for pions with |eta|<2.4 and pT>0.9 GeV. First search for a suitable pi+
    if ( itp1->pdgId() == 211 
	 && itp1->parentVertex().isNonnull() 
	 && abs(itp1->momentum().eta()) < 2.4 
	 && sqrt(itp1->momentum().perp2()) > 0.9) {
      if ( itp1->parentVertex()->nDaughterTracks() == 2 ) {
	TrackingVertexRef piCand1Vertex = itp1->parentVertex();	       
	//check trackingParticle pion for a Ks mother
	for (TrackingVertex::tp_iterator iTP1 = piCand1Vertex->sourceTracks_begin();
	     iTP1 != piCand1Vertex->sourceTracks_end(); ++iTP1) {
	  //iTP1 is a TrackingParticle
	  if ( (*iTP1)->pdgId()==310 ) {
	    //with a Ks mother found for the pi+, loop through trackingParticles again to find a pi-
	    for (TrackingParticleCollection::size_type j=0; j<tPCeff.size(); j++){
	      TrackingParticleRef tpr2(TPCollectionEff, j);
	      TrackingParticle* itp2=const_cast<TrackingParticle*>(tpr2.get());
	      
	      if ( itp2->pdgId() == -211 && itp2->parentVertex().isNonnull()  
		   && abs(itp2->momentum().eta()) < 2.4 
		   && sqrt(itp2->momentum().perp2()) > 0.9) {
		//check the pi+ and pi- have the same vertex
		if ( itp2->parentVertex() == itp1->parentVertex() ) {
		  TrackingVertexRef piCand2Vertex = itp2->parentVertex();	       
		  for (TrackingVertex::tp_iterator iTP2 = piCand2Vertex->sourceTracks_begin();
		       iTP2 != piCand2Vertex->sourceTracks_end(); ++iTP2) {
		    //iTP2 is a TrackingParticle
		    K0sGenEta = K0sGenpT = K0sGenR = 0.;
		    K0sGenStatus = 0;
		    if( (*iTP2)->pdgId() == 310 ) {
		      K0sGenpT = sqrt( (*iTP2)->momentum().perp2() );
		      K0sGenEta = (*iTP2)->momentum().eta();
		      K0sGenR = sqrt(itp2->vertex().perp2());
		      genK0s++;
		      if( trueK0s.size() > 0 ) {
			for( std::vector< pair<TrackingParticleRef, TrackingParticleRef> >::const_iterator iEffCheck = trueK0s.begin();
			     iEffCheck != trueK0s.end();
			     iEffCheck++) {
			  //if the parent vertices for the tracks are the same, then the generated Ks was found
			  if (itp1->parentVertex()==iEffCheck->first->parentVertex() &&
			      itp2->parentVertex()==iEffCheck->second->parentVertex())  {
			    realK0sFoundEff++;
			    K0sGenStatus = 1;
			    break;
			  }
			  else {
			    K0sGenStatus = 2;
			  }
			}
		      }
		      else {
			K0sGenStatus = 2;
		      }

		      // Check if the generated Ks tracks were found or not
		      // by searching the recoTracks list for a match to the trackingParticles

		      std::vector<std::pair<RefToBase<reco::Track>, double> > rt1;
		      std::vector<std::pair<RefToBase<reco::Track>, double> > rt2;
		      
		      if( simRecColl.find(tpr1) != simRecColl.end() ) {
			rt1 = (std::vector< std::pair<RefToBase<reco::Track>, double> >) simRecColl[tpr1];
			if(rt1.size() != 0) {
			  //First pion found
			  K0sPiEff[0] = 1;
			  edm::RefToBase<reco::Track> t1 = rt1.begin()->first;
			}
		      }
		      else {
			//First pion not found
			K0sPiEff[0] = 2;
		      }
		      
		      if( simRecColl.find(tpr2) != simRecColl.end() ) {
			rt2 = (std::vector< std::pair<RefToBase<reco::Track>, double> >) simRecColl[tpr2];
			if(rt2.size() != 0) {
			  //Second pion found
			  K0sPiEff[1] = 1;
			  edm::RefToBase<reco::Track> t2 = rt2.begin()->first;
			}
		      }
		      else {
			K0sPiEff[1] = 2;
		      }
		      //cout << "Status: " << K0sGenStatus << ", K0sPiEff[i]: " << K0sPiEff[0] << ", " << K0sPiEff[1] << endl;
		      if(K0sGenStatus == 1
			 && (K0sPiEff[0] == 2 || K0sPiEff[1] == 2)) {
			K0sGenStatus = 4;
			realK0sFoundEff--;
		      }
		      if(K0sGenStatus == 2
			 && (K0sPiEff[0] == 2 || K0sPiEff[1] == 2)) {
			K0sGenStatus = 3;
		      }
		      if(K0sPiEff[0] == 1 && K0sPiEff[1] == 1) {
			k0sTracksFound++;
		      }
		      //Fill Histograms
		      if(K0sGenR > 0.) {
			if(K0sGenStatus == 1) {
			  ksEffVsRHist->Fill(K0sGenR);
			}
			if((double) K0sGenStatus < 2.5) {			  
			  ksTkEffVsRHist->Fill(K0sGenR);
			}
			ksEffVsRHist_denom->Fill(K0sGenR);
		      }
		      if(abs(K0sGenEta) > 0.) {
			if(K0sGenStatus == 1) {
			  ksEffVsEtaHist->Fill(K0sGenEta);
			}
			if((double) K0sGenStatus < 2.5) {
			  ksTkEffVsEtaHist->Fill(K0sGenEta);
			}
			ksEffVsEtaHist_denom->Fill(K0sGenEta);
		      }
		      if(K0sGenpT > 0.) {
			if(K0sGenStatus == 1) {
			  ksEffVsPtHist->Fill(K0sGenpT);
			}
			if((double) K0sGenStatus < 2.5) {
			  ksTkEffVsPtHist->Fill(K0sGenpT);
			}
			ksEffVsPtHist_denom->Fill(K0sGenpT);
		      }
		    }
		  }
		}
	      }
	    }
	  }
	}
      }
    }
  }

  delete thePrimary;
}


void V0Validator::endJob() {
  //std::cout << "In endJob()" << std::endl;
  ksEffVsRHist->Divide(ksEffVsRHist_denom);
  ksEffVsEtaHist->Divide(ksEffVsEtaHist_denom);
  ksEffVsPtHist->Divide(ksEffVsPtHist_denom);
  ksTkEffVsRHist->Divide(ksEffVsRHist_denom);
  ksTkEffVsEtaHist->Divide(ksEffVsEtaHist_denom);
  ksTkEffVsPtHist->Divide(ksEffVsPtHist_denom);
  ksFakeVsRHist->Divide(ksFakeVsRHist_denom);
  ksFakeVsEtaHist->Divide(ksFakeVsEtaHist_denom);
  ksFakeVsPtHist->Divide(ksFakeVsPtHist_denom);
  ksTkFakeVsRHist->Divide(ksFakeVsRHist_denom);
  ksTkFakeVsEtaHist->Divide(ksFakeVsEtaHist_denom);
  ksTkFakeVsPtHist->Divide(ksFakeVsPtHist_denom);

  lamEffVsRHist->Divide(lamEffVsRHist_denom);
  lamEffVsEtaHist->Divide(lamEffVsEtaHist_denom);
  lamEffVsPtHist->Divide(lamEffVsPtHist_denom);
  lamTkEffVsRHist->Divide(lamEffVsRHist_denom);
  lamTkEffVsEtaHist->Divide(lamEffVsEtaHist_denom);
  lamTkEffVsPtHist->Divide(lamEffVsPtHist_denom);
  lamFakeVsRHist->Divide(lamFakeVsRHist_denom);
  lamFakeVsEtaHist->Divide(lamFakeVsEtaHist_denom);
  lamFakeVsPtHist->Divide(lamFakeVsPtHist_denom);
  lamTkFakeVsRHist->Divide(lamFakeVsRHist_denom);
  lamTkFakeVsEtaHist->Divide(lamFakeVsEtaHist_denom);
  lamTkFakeVsPtHist->Divide(lamFakeVsPtHist_denom);

  ksEffVsR = theDQMstore->book1D("KsEffVsR", ksEffVsRHist);
  ksEffVsEta = theDQMstore->book1D("KsEffVsEta", ksEffVsEtaHist);
  ksEffVsPt = theDQMstore->book1D("KsEffVsPt", ksEffVsPtHist);
  ksTkEffVsR = theDQMstore->book1D("KsTkEffVsR", ksTkEffVsRHist);
  ksTkEffVsEta = theDQMstore->book1D("KsTkEffVsEta", ksTkEffVsEtaHist);
  ksTkEffVsPt = theDQMstore->book1D("KsTkEffVsPt", ksTkEffVsPtHist);
  ksFakeVsR = theDQMstore->book1D("KsFakeVsR", ksFakeVsRHist);
  ksFakeVsEta = theDQMstore->book1D("KsFakeVsEta", ksFakeVsEtaHist);
  ksFakeVsPt = theDQMstore->book1D("KsFakeVsPt", ksFakeVsPtHist);
  ksTkFakeVsR = theDQMstore->book1D("KsTkFakeVsR", ksTkFakeVsRHist);
  ksTkFakeVsEta = theDQMstore->book1D("KsTkFakeVsEta", ksTkFakeVsEtaHist);
  ksTkFakeVsPt = theDQMstore->book1D("KsTkFakeVsPt", ksTkFakeVsPtHist);

  lamEffVsR = theDQMstore->book1D("LamEffVsR", lamEffVsRHist);
  lamEffVsEta = theDQMstore->book1D("LamEffVsEta", lamEffVsEtaHist);
  lamEffVsPt = theDQMstore->book1D("LamEffVsPt", lamEffVsPtHist);
  lamTkEffVsR = theDQMstore->book1D("LamTkEffVsR", lamTkEffVsRHist);
  lamTkEffVsEta = theDQMstore->book1D("LamTkEffVsEta", lamTkEffVsEtaHist);
  lamTkEffVsPt = theDQMstore->book1D("LamTkEffVsPt", lamTkEffVsPtHist);
  lamFakeVsR = theDQMstore->book1D("LamFakeVsR", lamFakeVsRHist);
  lamFakeVsEta = theDQMstore->book1D("LamFakeVsEta", lamFakeVsEtaHist);
  lamFakeVsPt = theDQMstore->book1D("LamFakeVsPt", lamFakeVsPtHist);
  lamTkFakeVsR = theDQMstore->book1D("LamTkFakeVsR", lamTkFakeVsRHist);
  lamTkFakeVsEta = theDQMstore->book1D("LamTkFakeVsEta", lamTkFakeVsEtaHist);
  lamTkFakeVsPt = theDQMstore->book1D("LamTkFakeVsPt", lamTkFakeVsPtHist);

  nKs = theDQMstore->book1D("nK0s", nKsHist);
  nLam = theDQMstore->book1D("nLam", nLamHist);

  theDQMstore->showDirStructure();
  theDQMstore->save(theDQMRootFileName);
}

//define this as a plug-in
//DEFINE_FWK_MODULE(V0Validator);
