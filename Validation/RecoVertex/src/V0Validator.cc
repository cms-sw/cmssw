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
//
//


#include "Validation/RecoVertex/interface/V0Validator.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

typedef std::vector<TrackingVertex> TrackingVertexCollection;
typedef edm::Ref<TrackingVertexCollection> TrackingVertexRef;
typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex > GenVertexRefVector;
typedef edm::RefVector<edm::HepMCProduct, HepMC::GenParticle > GenParticleRefVector;

V0Validator::V0Validator(const edm::ParameterSet& iConfig)
  : theDQMRootFileName(iConfig.getParameter<std::string>("DQMRootFileName"))
  , dirName(iConfig.getParameter<std::string>("dirName"))
  , recoRecoToSimCollectionToken_( consumes<reco::RecoToSimCollection>( edm::InputTag( std::string( "trackingParticleRecoTrackAsssociation" ) ) ) )
  , recoSimToRecoCollectionToken_( consumes<reco::SimToRecoCollection>( edm::InputTag( std::string( "trackingParticleRecoTrackAsssociation" ) ) ) )
  , trackingParticleCollection_Eff_Token_( consumes<TrackingParticleCollection>( edm::InputTag( std::string( "mix" )
											      , std::string( "MergedTrackTruth" )
												)
										 )
					   )
  , trackingParticleCollectionToken_( consumes<TrackingParticleCollection>( edm::InputTag( std::string( "mix" )
											 , std::string( "MergedTrackTruth" )
											   )
									    )
				      )
  , edmView_recoTrack_Token_( consumes< edm::View<reco::Track> >( edm::InputTag( std::string( "generalTracks" ) ) ) )
  , edmSimTrackContainerToken_( consumes<edm::SimTrackContainer>( edm::InputTag( std::string( "g4SimHits" ) ) ) )
  , edmSimVertexContainerToken_( consumes<edm::SimVertexContainer>( edm::InputTag( std::string( "g4SimHits" ) ) ) )
  , vec_recoVertex_Token_( consumes< std::vector<reco::Vertex> >( edm::InputTag( std::string( "offlinePrimaryVertices" ) ) ) )
  , recoVertexCompositeCandidateCollection_k0s_Token_( consumes<reco::VertexCompositeCandidateCollection>( iConfig.getParameter<edm::InputTag>( "kShortCollection" ) ) )
  , recoVertexCompositeCandidateCollection_lambda_Token_( consumes<reco::VertexCompositeCandidateCollection>( iConfig.getParameter<edm::InputTag>( "lambdaCollection" ) ) )
  , recoTrackToTrackingParticleAssociator_Token_( consumes<reco::TrackToTrackingParticleAssociator>(edm::InputTag("trackAssociatorByHits")) )
{

  genLam = genK0s = realLamFoundEff = realK0sFoundEff = lamCandFound = k0sCandFound = noTPforK0sCand = noTPforLamCand = realK0sFound = realLamFound = 0;
  
}

V0Validator::~V0Validator() {

}

void V0Validator::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const &, edm::EventSetup const &) {
  ibooker.cd();
  std::string subDirName = V0Validator::dirName + "/EffFakes";
  ibooker.setCurrentFolder(subDirName.c_str());

  ksEffVsR = ibooker.book1D("K0sEffVsR", 
			  "K^{0}_{S} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEta = ibooker.book1D("K0sEffVsEta",
			    "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPt = ibooker.book1D("K0sEffVsPt",
			   "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);;

  ksTkEffVsR = ibooker.book1D("K0sTkEffVsR", 
			  "K^{0}_{S} Tracking Efficiency vs #rho", 40, 0., 40.);
  ksTkEffVsEta = ibooker.book1D("K0sTkEffVsEta",
			    "K^{0}_{S} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  ksTkEffVsPt = ibooker.book1D("K0sTkEffVsPt",
			   "K^{0}_{S} Tracking Efficiency vs p_{T}", 70, 0., 20.);

  ksEffVsR_num = ibooker.book1D("K0sEffVsR_num", 
			  "K^{0}_{S} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEta_num = ibooker.book1D("K0sEffVsEta_num",
			    "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPt_num = ibooker.book1D("K0sEffVsPt_num",
			   "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);;

  ksTkEffVsR_num = ibooker.book1D("K0sTkEffVsR_num", 
			  "K^{0}_{S} Tracking Efficiency vs #rho", 40, 0., 40.);
  ksTkEffVsEta_num = ibooker.book1D("K0sTkEffVsEta_num",
			    "K^{0}_{S} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  ksTkEffVsPt_num = ibooker.book1D("K0sTkEffVsPt_num",
			   "K^{0}_{S} Tracking Efficiency vs p_{T}", 70, 0., 20.);;

  ksEffVsR_denom = ibooker.book1D("K0sEffVsR_denom", 
			  "K^{0}_{S} Efficiency vs #rho", 40, 0., 40.);
  ksEffVsEta_denom = ibooker.book1D("K0sEffVsEta_denom",
			    "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  ksEffVsPt_denom = ibooker.book1D("K0sEffVsPt_denom",
			   "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);;

  lamEffVsR = ibooker.book1D("LamEffVsR",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEta = ibooker.book1D("LamEffVsEta",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPt = ibooker.book1D("LamEffVsPt",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);

  lamTkEffVsR = ibooker.book1D("LamTkEffVsR",
			   "#Lambda^{0} TrackingEfficiency vs #rho", 40, 0., 40.);
  lamTkEffVsEta = ibooker.book1D("LamTkEffVsEta",
			     "#Lambda^{0} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  lamTkEffVsPt = ibooker.book1D("LamTkEffVsPt",
			    "#Lambda^{0} Tracking Efficiency vs p_{T}", 70, 0., 20.);

  lamEffVsR_num = ibooker.book1D("LamEffVsR_num",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEta_num = ibooker.book1D("LamEffVsEta_num",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPt_num = ibooker.book1D("LamEffVsPt_num",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);

  lamTkEffVsR_num = ibooker.book1D("LamTkEffVsR_num",
			   "#Lambda^{0} TrackingEfficiency vs #rho", 40, 0., 40.);
  lamTkEffVsEta_num = ibooker.book1D("LamTkEffVsEta_num",
			     "#Lambda^{0} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  lamTkEffVsPt_num = ibooker.book1D("LamTkEffVsPt_num",
			    "#Lambda^{0} Tracking Efficiency vs p_{T}", 70, 0., 20.);

  lamEffVsR_denom = ibooker.book1D("LamEffVsR_denom",
			   "#Lambda^{0} Efficiency vs #rho", 40, 0., 40.);
  lamEffVsEta_denom = ibooker.book1D("LamEffVsEta_denom",
			     "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  lamEffVsPt_denom = ibooker.book1D("LamEffVsPt_denom",
			    "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);

  ksFakeVsR = ibooker.book1D("K0sFakeVsR",
			   "K^{0}_{S} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEta = ibooker.book1D("K0sFakeVsEta",
			     "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPt = ibooker.book1D("K0sFakeVsPt",
			    "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);
  ksTkFakeVsR = ibooker.book1D("K0sTkFakeVsR",
			   "K^{0}_{S} Tracking Fake Rate vs #rho", 40, 0., 40.);
  ksTkFakeVsEta = ibooker.book1D("K0sTkFakeVsEta",
			     "K^{0}_{S} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  ksTkFakeVsPt = ibooker.book1D("K0sTkFakeVsPt",
			    "K^{0}_{S} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  ksFakeVsR_num = ibooker.book1D("K0sFakeVsR_num",
			   "K^{0}_{S} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEta_num = ibooker.book1D("K0sFakeVsEta_num",
			     "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPt_num = ibooker.book1D("K0sFakeVsPt_num",
			    "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);
  ksTkFakeVsR_num = ibooker.book1D("K0sTkFakeVsR_num",
			   "K^{0}_{S} Tracking Fake Rate vs #rho", 40, 0., 40.);
  ksTkFakeVsEta_num = ibooker.book1D("K0sTkFakeVsEta_num",
			     "K^{0}_{S} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  ksTkFakeVsPt_num = ibooker.book1D("K0sTkFakeVsPt_num",
			    "K^{0}_{S} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  ksFakeVsR_denom = ibooker.book1D("K0sFakeVsR_denom",
			   "K^{0}_{S} Fake Rate vs #rho", 40, 0., 40.);
  ksFakeVsEta_denom = ibooker.book1D("K0sFakeVsEta_denom",
			     "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  ksFakeVsPt_denom = ibooker.book1D("K0sFakeVsPt_denom",
			    "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);

  lamFakeVsR = ibooker.book1D("LamFakeVsR",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEta = ibooker.book1D("LamFakeVsEta",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPt = ibooker.book1D("LamFakeVsPt",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);
  lamTkFakeVsR = ibooker.book1D("LamTkFakeVsR",
			    "#Lambda^{0} Tracking Fake Rate vs #rho", 40, 0., 40.);
  lamTkFakeVsEta = ibooker.book1D("LamTkFakeVsEta",
			      "#Lambda^{0} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  lamTkFakeVsPt = ibooker.book1D("LamTkFakeVsPt",
			     "#Lambda^{0} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  lamFakeVsR_num = ibooker.book1D("LamFakeVsR_num",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEta_num = ibooker.book1D("LamFakeVsEta_num",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPt_num = ibooker.book1D("LamFakeVsPt_num",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);
  lamTkFakeVsR_num = ibooker.book1D("LamTkFakeVsR_num",
			    "#Lambda^{0} Tracking Fake Rate vs #rho", 40, 0., 40.);
  lamTkFakeVsEta_num = ibooker.book1D("LamTkFakeVsEta_num",
			      "#Lambda^{0} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  lamTkFakeVsPt_num = ibooker.book1D("LamTkFakeVsPt_num",
			     "#Lambda^{0} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  lamFakeVsR_denom = ibooker.book1D("LamFakeVsR_denom",
			    "#Lambda^{0} Fake Rate vs #rho", 40, 0., 40.);
  lamFakeVsEta_denom = ibooker.book1D("LamFakeVsEta_denom",
			      "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  lamFakeVsPt_denom = ibooker.book1D("LamFakeVsPt_denom",
			     "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);

  ibooker.cd();
  subDirName = dirName + "/Other";
  ibooker.setCurrentFolder(subDirName.c_str());

  nKs = ibooker.book1D("nK0s",
		     "Number of K^{0}_{S} found per event", 60, 0., 60.);
  nLam = ibooker.book1D("nLam",
		      "Number of #Lambda^{0} found per event", 60, 0., 60.);

  ksXResolution = ibooker.book1D("ksXResolution",
			       "Resolution of V0 decay vertex X coordinate", 50, 0., 50.);
  ksYResolution = ibooker.book1D("ksYResolution",
			       "Resolution of V0 decay vertex Y coordinate", 50, 0., 50.);
  ksZResolution = ibooker.book1D("ksZResolution",
			       "Resolution of V0 decay vertex Z coordinate", 50, 0., 50.);
  lamXResolution = ibooker.book1D("lamXResolution",
				"Resolution of V0 decay vertex X coordinate", 50, 0., 50.);
  lamYResolution = ibooker.book1D("lamYResolution",
				"Resolution of V0 decay vertex Y coordinate", 50, 0., 50.);
  lamZResolution = ibooker.book1D("lamZResolution",
				"Resolution of V0 decay vertex Z coordinate", 50, 0., 50.);
  ksAbsoluteDistResolution = ibooker.book1D("ksRResolution",
					  "Resolution of absolute distance from primary vertex to V0 vertex",
					  100, 0., 50.);
  lamAbsoluteDistResolution = ibooker.book1D("lamRResolution",
					   "Resolution of absolute distance from primary vertex to V0 vertex",
					   100, 0., 50.);

  ksCandStatus = ibooker.book1D("ksCandStatus",
			  "Fake type by cand status",
			  10, 0., 10.);
  lamCandStatus = ibooker.book1D("ksCandStatus",
			  "Fake type by cand status",
			  10, 0., 10.);

  double minKsMass = 0.49767 - 0.07;
  double maxKsMass = 0.49767 + 0.07;
  double minLamMass = 1.1156 - 0.05;
  double maxLamMass = 1.1156 + 0.05;
  int ksMassNbins = 100;
  double ksMassXmin = minKsMass;
  double ksMassXmax = maxKsMass;
  int lamMassNbins = 100;
  double lamMassXmin = minLamMass;
  double lamMassXmax = maxLamMass;

  fakeKsMass = ibooker.book1D("ksMassFake",
			     "Mass of fake K0S",
			     ksMassNbins, minKsMass, maxKsMass);
  goodKsMass = ibooker.book1D("ksMassGood",
			     "Mass of good reco K0S",
			     ksMassNbins, minKsMass, maxKsMass);
  fakeLamMass = ibooker.book1D("lamMassFake",
			      "Mass of fake Lambda",
			      lamMassNbins, minLamMass, maxLamMass);
  goodLamMass = ibooker.book1D("lamMassGood",
			      "Mass of good Lambda",
			      lamMassNbins, minLamMass, maxLamMass);

  ksMassAll = ibooker.book1D("ksMassAll",
				  "Invariant mass of all K0S",
				  ksMassNbins, ksMassXmin, ksMassXmax);
  lamMassAll = ibooker.book1D("lamMassAll",
				   "Invariant mass of all #Lambda^{0}",
				   lamMassNbins, lamMassXmin, lamMassXmax);

  ksFakeDauRadDist = ibooker.book1D("radDistFakeKs",
				   "Production radius of daughter particle of Ks fake",
				   100, 0., 15.);
  lamFakeDauRadDist = ibooker.book1D("radDistFakeLam",
				    "Production radius of daughter particle of Lam fake",
				    100, 0., 15.);
}

void V0Validator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  using std::cout;
  using std::endl;
  using namespace edm;
  using namespace std;

  // Get event setup info, B-field and tracker geometry
  ESHandle<MagneticField> bFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  ESHandle<GlobalTrackingGeometry> globTkGeomHandle;
  iSetup.get<GlobalTrackingGeometryRecord>().get(globTkGeomHandle);

  // Make matching collections
  Handle<reco::RecoToSimCollection > recotosimCollectionH;
  iEvent.getByToken( recoRecoToSimCollectionToken_, recotosimCollectionH );
  
  Handle<reco::SimToRecoCollection> simtorecoCollectionH;
  iEvent.getByToken( recoSimToRecoCollectionToken_, simtorecoCollectionH );

  edm::Handle<TrackingParticleCollection>  TPCollectionEff ;
  iEvent.getByToken( trackingParticleCollection_Eff_Token_, TPCollectionEff );
  const TrackingParticleCollection tPCeff = *( TPCollectionEff.product() );

  edm::Handle<reco::TrackToTrackingParticleAssociator> associatorByHits;
  iEvent.getByToken(recoTrackToTrackingParticleAssociator_Token_, associatorByHits);

  // Get tracks
  Handle< View<reco::Track> > trackCollectionH;
  iEvent.getByToken( edmView_recoTrack_Token_, trackCollectionH );

  Handle<SimTrackContainer> simTrackCollection;
  iEvent.getByToken( edmSimTrackContainerToken_, simTrackCollection );
  const SimTrackContainer simTC = *(simTrackCollection.product());

  Handle<SimVertexContainer> simVertexCollection;
  iEvent.getByToken( edmSimVertexContainerToken_, simVertexCollection );
  const SimVertexContainer simVC = *(simVertexCollection.product());

  //Get tracking particles
  //  -->tracks
  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  iEvent.getByToken( trackingParticleCollectionToken_, TPCollectionH );
  const View<reco::Track>  tC = *( trackCollectionH.product() );

  // Select the primary vertex, create a new reco::Vertex to hold it
  edm::Handle< std::vector<reco::Vertex> > primaryVtxCollectionH;
  iEvent.getByToken( vec_recoVertex_Token_, primaryVtxCollectionH );
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

  //get the V0s;   
  edm::Handle<reco::VertexCompositeCandidateCollection> k0sCollection;
  edm::Handle<reco::VertexCompositeCandidateCollection> lambdaCollection;
  iEvent.getByToken( recoVertexCompositeCandidateCollection_k0s_Token_, k0sCollection );
  iEvent.getByToken( recoVertexCompositeCandidateCollection_lambda_Token_, lambdaCollection );

  //make vector of pair of trackingParticles to hold good V0 candidates
  std::vector< pair<TrackingParticleRef, TrackingParticleRef> > trueK0s;
  std::vector< pair<TrackingParticleRef, TrackingParticleRef> > trueLams;
  std::vector<double> trueKsMasses;
  std::vector<double> trueLamMasses;

  //////////////////////////////
  // Do fake rate calculation //
  //////////////////////////////

  // Kshorts
  double numK0sFound = 0.;
  double mass = 0.;
  std::vector<double> radDist;
  if ( k0sCollection->size() > 0 ) {
    vector<reco::TrackRef> theDaughterTracks;
    for( reco::VertexCompositeCandidateCollection::const_iterator iK0s = k0sCollection->begin();
	 iK0s != k0sCollection->end();
	 iK0s++) {
      // Fill mass of all K0S
      ksMassAll->Fill( iK0s->mass() );
      // Fill values to be histogrammed
      K0sCandpT = (sqrt( iK0s->momentum().perp2() ));
      K0sCandEta = iK0s->momentum().eta();
      K0sCandR = (sqrt( iK0s->vertex().perp2() ));
      K0sCandStatus = 0;
      mass = iK0s->mass();

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
        
	if(recotosimCollectionH->find(track) != recotosimCollectionH->end()) {
	  tp = (*recotosimCollectionH)[track];
	  if (tp.size() != 0) {
	    K0sPiCandStatus[i] = 1;
	    tpref = tp.begin()->first;

	    if( simtorecoCollectionH->find(tpref) == simtorecoCollectionH->end() ) {
	      K0sPiCandStatus[i] = 3;
	    }
	    TrackingVertexRef parentVertex = tpref->parentVertex();
	    if(parentVertex.isNonnull()) radDist.push_back(parentVertex->position().R());
	     
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
			trueKsMasses.push_back(mass);
		      }
		      else {
			K0sCandStatus = 2;
			if( (*iTP)->pdgId() == 3122 ) {
			  K0sCandStatus = 7;
			}
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
	ksFakeVsR_num->Fill(K0sCandR);
	ksFakeVsEta_num->Fill(K0sCandEta);
	ksFakeVsPt_num->Fill(K0sCandpT);
	ksCandStatus->Fill((float) K0sCandStatus);
	fakeKsMass->Fill(mass);
	for( unsigned int ndx = 0; ndx < radDist.size(); ndx++ ) {
	  ksFakeDauRadDist->Fill(radDist[ndx]);
	}
      }
      if( K0sCandStatus == 5 ) {
	ksTkFakeVsR_num->Fill(K0sCandR);
	ksTkFakeVsEta_num->Fill(K0sCandEta);
	ksTkFakeVsPt_num->Fill(K0sCandpT);
      }
      ksFakeVsR_denom->Fill(K0sCandR);
      ksFakeVsEta_denom->Fill(K0sCandEta);
      ksFakeVsPt_denom->Fill(K0sCandpT);
    }
  }
  nKs->Fill( (float) numK0sFound );
  numK0sFound = 0.;

  double numLamFound = 0.;
  mass = 0.;
  radDist.clear();
  // Lambdas
  if ( lambdaCollection->size() > 0 ) {
    vector<reco::TrackRef> theDaughterTracks;
    for( reco::VertexCompositeCandidateCollection::const_iterator iLam = lambdaCollection->begin();
	 iLam != lambdaCollection->end();
	 iLam++) {
      // Fill mass plot with ALL lambdas
      lamMassAll->Fill( iLam->mass() );
      // Fill values to be histogrammed
      LamCandpT = (sqrt( iLam->momentum().perp2() ));
      LamCandEta = iLam->momentum().eta();
      LamCandR = (sqrt( iLam->vertex().perp2() ));
      LamCandStatus = 0;
      mass = iLam->mass();
      
      //cout << "Lam daughter tracks" << endl;
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
	
	if(recotosimCollectionH->find(track) != recotosimCollectionH->end()) {
	  tp = (*recotosimCollectionH)[track];
	  if (tp.size() != 0) {
	    LamPiCandStatus[i] = 1;
	    tpref = tp.begin()->first;
	    if( simtorecoCollectionH->find(tpref) == simtorecoCollectionH->end() ) {
	      LamPiCandStatus[i] = 3;
	    }
	    TrackingVertexRef parentVertex = tpref->parentVertex();
	    if( parentVertex.isNonnull() ) radDist.push_back(parentVertex->position().R());
	     
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
			trueLamMasses.push_back(mass);
		      }
		      else {
			LamCandStatus = 2;
			if( abs((*iTP)->pdgId() ) == 310 ) {
			  LamCandStatus = 7;
			}
		      }
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
	lamFakeVsR_num->Fill(LamCandR);
	lamFakeVsEta_num->Fill(LamCandEta);
	lamFakeVsPt_num->Fill(LamCandpT);
	lamCandStatus->Fill((float) LamCandStatus);
	fakeLamMass->Fill(mass);
	for( unsigned int ndx = 0; ndx < radDist.size(); ndx++ ) {
	  lamFakeDauRadDist->Fill(radDist[ndx]);
	}
      }
      if( K0sCandStatus == 5 ) {
	lamTkFakeVsR_num->Fill(LamCandR);
	lamTkFakeVsEta_num->Fill(LamCandEta);
	lamTkFakeVsPt_num->Fill(LamCandpT);
      }
      lamFakeVsR_denom->Fill(LamCandR);
      lamFakeVsEta_denom->Fill(LamCandEta);
      lamFakeVsPt_denom->Fill(LamCandpT);
    }
  }
  nLam->Fill( (double) numLamFound );
  numLamFound = 0.;


  ///////////////////////////////
  // Do efficiency calculation //
  ///////////////////////////////
  // Lambdas
  for(TrackingParticleCollection::size_type i = 0; i < tPCeff.size(); i++) {
    TrackingParticleRef tpr1(TPCollectionEff, i);
    const TrackingParticle* itp1 = tpr1.get();
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
	    //	     ----->>>>>>Keep going here
	    for(TrackingParticleCollection::size_type j=0;
		j < tPCeff.size();
		j++) {
	      TrackingParticleRef tpr2(TPCollectionEff, j);
	      const TrackingParticle* itp2 = tpr2.get();
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
			int loop_1 = 0;
			for(std::vector< pair<TrackingParticleRef, TrackingParticleRef> >::const_iterator iEffCheck = trueLams.begin();
			    iEffCheck != trueLams.end();
			    iEffCheck++) {
			  if( itp1->parentVertex() == iEffCheck->first->parentVertex()
			      && itp2->parentVertex() == iEffCheck->second->parentVertex() ) {
			    realLamFoundEff++;
			    //V0Producer found the generated Lambda
			    LamGenStatus = 1;
			    goodLamMass->Fill(trueLamMasses[loop_1]);
			    break;
			  }
			  else {
			    //V0Producer didn't find the generated Lambda
			    LamGenStatus = 2;
			  }
			  loop_1++;
			}
		      }
		      else {
			//No V0 cand found, so V0Producer didn't find the generated Lambda
			LamGenStatus = 2;
		      }
		      std::vector< std::pair<RefToBase<reco::Track>, double> > rt1;
		      std::vector< std::pair<RefToBase<reco::Track>, double> > rt2;
		      
		      if( simtorecoCollectionH->find(tpr1) != simtorecoCollectionH->end() ) {
			rt1 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) (*simtorecoCollectionH)[tpr1];
			if(rt1.size() != 0) {
			  LamPiEff[0] = 1; //Found the first daughter track
			  edm::RefToBase<reco::Track> t1 = rt1.begin()->first;
			}
		      }
		      else {
			LamPiEff[0] = 2;//First daughter not found
		      }
		      if( (simtorecoCollectionH->find(tpr2) != simtorecoCollectionH->end()) ) {
			rt2 = (std::vector<std::pair<RefToBase<reco::Track>, double> >) (*simtorecoCollectionH)[tpr2];
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
		      // Fill histograms
		      if(LamGenR > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsR_num->Fill(LamGenR);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsR_num->Fill(LamGenR);
			}
			lamEffVsR_denom->Fill(LamGenR);
		      }
		      if(abs(LamGenEta) > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsEta_num->Fill(LamGenEta);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsEta_num->Fill(LamGenEta);
			}
			lamEffVsEta_denom->Fill(LamGenEta);
		      }
		      if(LamGenpT > 0.) {
			if(LamGenStatus == 1) {
			  lamEffVsPt_num->Fill(LamGenpT);
			}
			if((double) LamGenStatus < 2.5) {
			  lamTkEffVsPt_num->Fill(LamGenpT);
			}
			lamEffVsPt_denom->Fill(LamGenpT);
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

  for (TrackingParticleCollection::size_type i=0; i<tPCeff.size(); i++){
    TrackingParticleRef tpr1(TPCollectionEff, i);
    const TrackingParticle* itp1 = tpr1.get();
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
	      const TrackingParticle* itp2 = tpr2.get();
	      
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
		      int loop_2 = 0;
		      if( trueK0s.size() > 0 ) {
			for( std::vector< pair<TrackingParticleRef, TrackingParticleRef> >::const_iterator iEffCheck = trueK0s.begin();
			     iEffCheck != trueK0s.end();
			     iEffCheck++) {
			  //if the parent vertices for the tracks are the same, then the generated Ks was found
			  if (itp1->parentVertex()==iEffCheck->first->parentVertex() &&
			      itp2->parentVertex()==iEffCheck->second->parentVertex())  {
			    realK0sFoundEff++;
			    K0sGenStatus = 1;
			    goodKsMass->Fill(trueKsMasses[loop_2]);
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
		      
		      if( simtorecoCollectionH->find(tpr1) != simtorecoCollectionH->end() ) {
			rt1 = (std::vector< std::pair<RefToBase<reco::Track>, double> >) (*simtorecoCollectionH)[tpr1];
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
		      
		      if( simtorecoCollectionH->find(tpr2) != simtorecoCollectionH->end() ) {
			rt2 = (std::vector< std::pair<RefToBase<reco::Track>, double> >) (*simtorecoCollectionH)[tpr2];
			if(rt2.size() != 0) {
			  //Second pion found
			  K0sPiEff[1] = 1;
			  edm::RefToBase<reco::Track> t2 = rt2.begin()->first;
			}
		      }
		      else {
			K0sPiEff[1] = 2;
		      }
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
			  ksEffVsR_num->Fill(K0sGenR);
			}
			if((double) K0sGenStatus < 2.5) {			  
			  ksTkEffVsR_num->Fill(K0sGenR);
			}
			ksEffVsR_denom->Fill(K0sGenR);
		      }
		      if(abs(K0sGenEta) > 0.) {
			if(K0sGenStatus == 1) {
			  ksEffVsEta_num->Fill(K0sGenEta);
			}
			if((double) K0sGenStatus < 2.5) {
			  ksTkEffVsEta_num->Fill(K0sGenEta);
			}
			ksEffVsEta_denom->Fill(K0sGenEta);
		      }
		      if(K0sGenpT > 0.) {
			if(K0sGenStatus == 1) {
			  ksEffVsPt_num->Fill(K0sGenpT);
			}
			if((double) K0sGenStatus < 2.5) {
			  ksTkEffVsPt_num->Fill(K0sGenpT);
			}
			ksEffVsPt_denom->Fill(K0sGenpT);
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

//define this as a plug-in
//DEFINE_FWK_MODULE(V0Validator);
