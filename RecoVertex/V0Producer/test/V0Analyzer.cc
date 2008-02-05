// -*- C++ -*-
//
// Package:    V0Analyzer
// Class:      V0Analyzer
// 
/**\class V0Analyzer V0Analyzer.cc RecoVertex/V0Producer/test/V0Analyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Tue May 22 23:54:16 CEST 2007
// $Id: V0Analyzer.cc,v 1.2 2008/02/04 21:54:58 drell Exp $
//
//

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

//#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/VolumeBasedEngine/interface/VolumeBasedMagneticField.h"
#include "DataFormats/V0Candidate/interface/V0Candidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"

//#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
//#include "Geometry/CommonDetUnit/interface/GeomDet.h"
//#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "TH1.h"
#include "TH2.h"
#include "TFile.h"
#include <fstream>
//
// class declaration
//

class V0Analyzer : public edm::EDAnalyzer {
   public:
      explicit V0Analyzer(const edm::ParameterSet&);
      ~V0Analyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  std::string algoLabel;
  std::string recoAlgoLabel;
  std::string SimTkLabel;
  std::string SimVtxLabel;
  std::string outputFile;
  std::string HistoFileName;
  std::string V0CollectionName;

  TFile* theHistoFile;
  TH1D* myKshortMassHisto;
  TH1D* myDecayRadiusHisto;
  TH1D* myRefitKshortMassHisto;
  TH1D* myRefitDecayRadiusHisto;
  TH1D* myNativeParticleMassHisto;
  TH1D* myEtaEfficiencyHisto;
  TH1D* mySimEtaHisto;
  TH1D* myRhoEfficiencyHisto;
  TH1D* mySimRhoHisto;
  TH1D* myKshortPtHisto;
  TH1D* myImpactParameterHisto;
  TH1D* myImpactParameterHisto2;
  TH1D* myNumSimKshortsHisto;
  TH1D* myNumRecoKshortsHisto;
  TH1D* myInnermostHitDistanceHisto;

  // Histograms for figuring out the best cuts
  TH1D* step1massHisto;
  TH1D* step2massHisto;
  TH1D* step3massHisto;

  TH1D* vertexChi2Histo;
  TH1D* rVtxHisto1;
  TH1D* vtxSigHisto1;
  TH1D* rVtxHisto2;
  TH1D* vtxSigHisto2;



      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//

const double piMassSq = 0.019479101;

//
// static data member definitions
//

//
// constructors and destructor
//
V0Analyzer::V0Analyzer(const edm::ParameterSet& iConfig):
  algoLabel(iConfig.getUntrackedParameter("recoAlgorithm",
					  std::string("ctfV0Prod"))),
  recoAlgoLabel(iConfig.getUntrackedParameter("trackingAlgo",
			       std::string("ctfWithMaterialTracks"))),
  SimTkLabel(iConfig.getUntrackedParameter("moduleLabelTk",
					 std::string("g4SimHits"))), 
  SimVtxLabel(iConfig.getUntrackedParameter("moduleLabelVtx",
					    std::string("g4SimHits"))),
  HistoFileName(iConfig.getUntrackedParameter("histoFileName",
			       std::string("vtxAnalyzerHistos.root"))), 
  V0CollectionName(iConfig.getUntrackedParameter("v0CollectionName",
						 std::string("Kshort"))) {

   //now do what ever initialization is needed
  theHistoFile = 0;
}


V0Analyzer::~V0Analyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//


// ------------ method called once each job just before starting event loop  ------------
void V0Analyzer::beginJob(const edm::EventSetup&) {

  using std::string;

  theHistoFile = new TFile(HistoFileName.c_str(), "RECREATE");

  string algo(algoLabel);
  string vernum("160");
  algo += vernum;
  string refit("Refitted");
  string native("Native");

  string massHistoLabelLong(algo + string(" Invariant Mass of K^{0}_{s}"));
  string massHistoLabelShort(vernum + string("K0sInvarMass"));

  string decRadHistoLabelLong(algo 
			      + string(" Distance of Decay Vtx from z-axis"));
  string decRadHistoLabelShort(vernum + string("K0sDecayRad"));

  string etaEffHistoLabelLong(algo + 
			      string(" Reconstruction Efficiency vs. #eta"));
  string etaEffHistoLabelShort(vernum + string("EffVsEta"));

  string rhoEffHistoLabelLong(algo +
			      string(" Reconstruction Efficiency vs. #rho"));
  string rhoEffHistoLabelShort(vernum + string("EffVsRho"));

  string simRhoHistoLabelLong(algo +
			      string(" Simulated K^{0}_{s} #rho"));
  string simRhoHistoLabelShort(vernum + string("SimRho"));

  string simEtaHistoLabelLong(algo +
			      string(" Simulated K^{0}_{s} #eta"));
  string simEtaHistoLabelShort(vernum + string("SimEta"));

  string kShortPtHistoLabelLong(algo +
				string(" K^{0}_{s} P_{t}"));
  string kShortPtHistoLabelShort(vernum + string("K0sPt"));

  string impactParamHistoLabelLong(algo+
				   string(" Impact parameter of reco tracks"));
  string impactParamHistoLabelShort(vernum + string("d0"));

  string impactParamHisto2LabelLong(algo+
			    string(" distance from beam line to reco tracks"));
  string impactParamHisto2LabelShort(vernum + string("d0Radial"));

  string numSimKshortsHistoLabelLong(algo+
				     string(" number of simulated Kshorts"));
  string numSimKshortsHistoLabelShort(vernum + string("numSimK0s"));

  string numRecoKshortsHistoLabelLong(algo+
				   string(" number of reconstructed Kshorts")); 
  string numRecoKshortsHistoLabelShort(vernum + string("numRecoK0s"));

  string innermostHitDistanceHistoLabelLong(algo+
		  string(" distance between innermost hits of track pairs"));
  string innermostHitDistanceHistoLabelShort(vernum + string("hitDist"));

  //-----------------------------------------

  string s1massHistoLabelLong(algo+
			      string(" m_{#pi #pi}, all tracks"));
  string s1massHistoLabelShort(vernum + string("mPiPiS1"));

  string s2massHistoLabelLong(algo+
		    string(" m_{#pi #pi}, R_{vtx} > 0.1, #chi^2 < 1"));
  string s2massHistoLabelShort(vernum + string("mPiPiS2"));

  string s3massHistoLabelLong(algo+
			      string(" m_{#pi #pi}, all cuts implemented"));
  string s3massHistoLabelShort(vernum + string("mPiPiS3"));

  string vertexChi2HistoLabelLong(algo+
				  string(" vertex #chi^{2}"));
  string vertexChi2HistoLabelShort(vernum + string("vtxChi2"));

  string rVtxHistoLabelLong(algo+
			    string(" r_{vtx} of V^{0} decay - radial"));
  string rVtxHistoLabelShort(vernum + string("rVtxRadial"));

  string vtxSigHistoLabelLong(algo+
			      string(" V^{0} vertex significance - radial"));
  string vtxSigHistoLabelShort(vernum + string("#sigma rVtxRadial"));

  string rVtxHistoLabel2Long(algo+
			    string(" r_{vtx} of V^{0} decay"));
  string rVtxHistoLabel2Short(vernum + string("rVtxSpherical"));

  string vtxSigHistoLabel2Long(algo+
			      string(" V^{0} vertex significance"));
  string vtxSigHistoLabel2Short(vernum + string("#sigma rVtxSpherical"));

  step1massHisto = new TH1D(s1massHistoLabelShort.c_str(),
			    s1massHistoLabelLong.c_str(),
			    100, 0., 2.);
  step2massHisto = new TH1D(s2massHistoLabelShort.c_str(),
			    s2massHistoLabelLong.c_str(),
			    100, 0., 2.);
  step3massHisto = new TH1D(s3massHistoLabelShort.c_str(),
			    s3massHistoLabelLong.c_str(),
			    100, 0., 2.);

  vertexChi2Histo = new TH1D(vertexChi2HistoLabelShort.c_str(),
			     vertexChi2HistoLabelLong.c_str(),
			     100, 0., 20.);
  rVtxHisto1 = new TH1D(rVtxHistoLabelShort.c_str(),
		       rVtxHistoLabelLong.c_str(),
		       100, 0., 1.);
  vtxSigHisto1 = new TH1D(vtxSigHistoLabelShort.c_str(),
			 vtxSigHistoLabelLong.c_str(),
			 100, 0., 100.);
  rVtxHisto2 = new TH1D(rVtxHistoLabel2Short.c_str(),
		       rVtxHistoLabel2Long.c_str(),
		       100, 0., 1.);
  vtxSigHisto2 = new TH1D(vtxSigHistoLabel2Short.c_str(),
			 vtxSigHistoLabel2Long.c_str(),
			 100, 0., 100.);


  myKshortMassHisto = new TH1D(massHistoLabelShort.c_str(), 
			       massHistoLabelLong.c_str(),
			       100, 0.3, 0.7);
  myDecayRadiusHisto = new TH1D(decRadHistoLabelShort.c_str(), 
				decRadHistoLabelLong.c_str(),
				100, 0., 40.);
  myRefitKshortMassHisto = new TH1D((refit+massHistoLabelShort).c_str(), 
				    (refit+massHistoLabelLong).c_str(),
				    100, 0.3, 0.7);
  myRefitDecayRadiusHisto = new TH1D((refit+decRadHistoLabelShort).c_str(), 
				     (refit+decRadHistoLabelLong).c_str(),
				     100, 0., 40.);
  myNativeParticleMassHisto = new TH1D((native+massHistoLabelShort).c_str(),
				       (refit+massHistoLabelLong).c_str(),
				       300, 0.3, 1.8);
  myEtaEfficiencyHisto = new TH1D(etaEffHistoLabelShort.c_str(),
				  etaEffHistoLabelLong.c_str(),
				  40, -2.5, 2.5);
  mySimEtaHisto = new TH1D(simEtaHistoLabelShort.c_str(),
			   simEtaHistoLabelLong.c_str(),
			   40, -2.5, 2.5);
  myRhoEfficiencyHisto = new TH1D(rhoEffHistoLabelShort.c_str(),
				  rhoEffHistoLabelLong.c_str(),
				  60, 0., 60.);
  mySimRhoHisto = new TH1D(simRhoHistoLabelShort.c_str(),
			   simRhoHistoLabelLong.c_str(),
			   60, 0., 60.);
  myKshortPtHisto = new TH1D(kShortPtHistoLabelShort.c_str(),
			    kShortPtHistoLabelLong.c_str(),
			    100, 0., 100.);
  myImpactParameterHisto = new TH1D(impactParamHistoLabelShort.c_str(),
				    impactParamHistoLabelLong.c_str(),
				    100, 0., 10.);
  myImpactParameterHisto2 = new TH1D(impactParamHisto2LabelShort.c_str(),
				    impactParamHisto2LabelLong.c_str(),
				    100, 0., 10.);
  myNumSimKshortsHisto = new TH1D(numSimKshortsHistoLabelShort.c_str(),
				  numSimKshortsHistoLabelLong.c_str(),
				  100, 0., 100.);
  myNumRecoKshortsHisto = new TH1D(numRecoKshortsHistoLabelShort.c_str(),
				   numRecoKshortsHistoLabelLong.c_str(),
				   100, 0., 100.);
  myInnermostHitDistanceHisto = 
    new TH1D(innermostHitDistanceHistoLabelShort.c_str(),
	     innermostHitDistanceHistoLabelLong.c_str(),
	     100, 0., 10.);

  myEtaEfficiencyHisto->Sumw2();
  mySimEtaHisto->Sumw2();
  myRhoEfficiencyHisto->Sumw2();
  mySimRhoHisto->Sumw2();

}

// ------------ method called to for each event  ------------
void V0Analyzer::analyze(const edm::Event& iEvent, 
			     const edm::EventSetup& iSetup) {
  std::cout << std::endl << "@@@In module..." << std::endl;
  std::cout << "Creating HANDLES..." << std::endl;
  using namespace edm;
  //Handle<reco::VertexCollection> theVtxHandle;
  Handle< std::vector<reco::Vertex> > theVtxHandle;
  Handle< std::vector<reco::V0Candidate> > theCandHand;
  Handle<SimTrackContainer> SimTk;
  Handle<SimVertexContainer> SimVtx;
  Handle<reco::TrackCollection> RecoTk;

  ESHandle<MagneticField> bFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(bFieldHandle);
  
  std::cout << "Getting by label..." << std::endl;
  iEvent.getByLabel(algoLabel, V0CollectionName, theCandHand);
  iEvent.getByLabel(SimTkLabel, SimTk);
  iEvent.getByLabel(SimVtxLabel, SimVtx);
  iEvent.getByLabel(recoAlgoLabel, RecoTk);

  std::cout << "Creating and filling vectors..." << std::endl;
  std::vector<reco::Track> theRecoTracks;
  std::vector<SimVertex> theSimVerts;
  std::vector<SimTrack> theSimTracks;
  std::vector<reco::V0Candidate> theKshorts;
  theRecoTracks.insert( theRecoTracks.end(), RecoTk->begin(), RecoTk->end() );
  theSimVerts.insert( theSimVerts.end(), SimVtx->begin(), SimVtx->end() );
  theSimTracks.insert( theSimTracks.end(), SimTk->begin(), SimTk->end() );
  theKshorts.insert( theKshorts.end(), theCandHand->begin(),
		     theCandHand->end() );

  double numSimKshorts = 0.;
  for(unsigned int ndx1 = 0; ndx1 < theSimTracks.size(); ndx1++) {
    if(theSimTracks[ndx1].type() == 310) {
      numSimKshorts += 1.;
    }
  }

  myNumSimKshortsHisto->Fill(numSimKshorts, 1.);

  std::cout << "Found " << theKshorts.size() << " K0s events."
	    << std::endl;

  double simRad = sqrt(theSimVerts[1].position().perp2());
  double simCosTheta = theSimVerts[1].position().z()
    / theSimVerts[1].position().mag();
  double simEta = -log( tan( acos( simCosTheta)/2. ) );

  mySimRhoHisto->Fill(simRad, 1.);
  mySimEtaHisto->Fill(simEta, 1.);

  
  bool hasKshort = false;
  bool hasLambda = false;
  bool hasLambdaBar = false;
  std::cout << theKshorts.size() << std::endl;

  if( V0CollectionName == std::string("Kshort") ) {
    myNumRecoKshortsHisto->Fill( (double) theKshorts.size(), 1. );
  }

  for(unsigned int ksndx_ = 0; ksndx_ < theKshorts.size(); ksndx_++) {
    myNativeParticleMassHisto->Fill( theKshorts[ksndx_].mass() );
  }

  for(unsigned int ksndx = 0; ksndx < theKshorts.size(); ksndx++) {
    std::vector<reco::RecoChargedCandidate> v0daughters;
    std::vector<reco::TrackRef> theDaughterTracks;

    for (unsigned int i = 0; i < theKshorts[ksndx].numberOfDaughters(); i++) {
      v0daughters.push_back( *(dynamic_cast<reco::RecoChargedCandidate *> 
			       (theKshorts[ksndx].daughter(i))) );
    }

    for(unsigned int j = 0; j < v0daughters.size(); j++) {
      theDaughterTracks.push_back(v0daughters[j].track());
    }

    reco::TrackBase::Point beamSpot(0,0,0);

    for(unsigned int k = 0; k < theDaughterTracks.size(); k++) {
      myImpactParameterHisto->Fill( sqrt( theDaughterTracks[k]->dxy( beamSpot )
				       * theDaughterTracks[k]->dxy( beamSpot )
				       + theDaughterTracks[k]->dsz( beamSpot )
				       * theDaughterTracks[k]->dsz( beamSpot )),
				    1.);
    }
    
    reco::Vertex k0s = theKshorts[ksndx].vertex();
    myKshortPtHisto->Fill(sqrt(theKshorts[ksndx].momentum().x() *
			       theKshorts[ksndx].momentum().x() +
			       theKshorts[ksndx].momentum().y() *
			       theKshorts[ksndx].momentum().y()), 1.);
    if(theKshorts[ksndx].pdgId() == 310) {
      hasKshort = true;
    }
    if(theKshorts[ksndx].pdgId() == 3122) {
      hasLambda = true;
    }
    if(theKshorts[ksndx].pdgId() == -3122) {
      hasLambdaBar = true;
    }
    std::cout << "$#@#$: " << hasKshort << " " << hasLambda << " "
	      << hasLambdaBar << " size: " << theKshorts.size() << std::endl;
    std::cout << "@@@ MASS: " << theKshorts[ksndx].mass() << std::endl;

    //std::cout << "tracksSize()=" << k0s.tracksSize() << std::endl;
    //std::cout << "hasRefittedTracks()="<< k0s.hasRefittedTracks() << std::endl;
    double decayRad = sqrt(k0s.x()*k0s.x() + k0s.y()*k0s.y());
    myDecayRadiusHisto->Fill(decayRad);
    
    GlobalPoint vtxPos(k0s.x(), k0s.y(), k0s.z());
    double recoRad = vtxPos.perp();
    double recoCosTheta = vtxPos.z() / vtxPos.mag();
    double recoEta = -log( tan( acos( recoCosTheta )/2. ) );

    double x_ = k0s.x();
    double y_ = k0s.y();
    double z_ = k0s.z();
    double sig00 = k0s.covariance(0,0);
    double sig11 = k0s.covariance(1,1);
    double sig22 = k0s.covariance(2,2);
    double sig01 = k0s.covariance(0,1);
    double sig02 = k0s.covariance(0,2);
    double sig12 = k0s.covariance(1,2);

    double vtxRSph = vtxPos.mag();
    double vtxR = vtxPos.perp();
    double vtxChi2 = k0s.chi2();
    double vtxErrorSph =
      sqrt( sig00*(x_*x_) + sig11*(y_*y_) + sig22*(z_*z_)
	    + 2*(sig01*(x_*y_) + sig02*(x_*z_) + sig12*(y_*z_)) ) 
      / vtxRSph;
    double vtxError =
      sqrt( sig00*(x_*x_) + sig11*(y_*y_)
	    + 2*sig01*(x_*y_) ) / vtxR;
    double vtxSigSph = vtxRSph / vtxErrorSph;
    double vtxSig = vtxR / vtxError;

    vertexChi2Histo->Fill(vtxChi2, 1.);

    step1massHisto->Fill(theKshorts[ksndx].mass(), 1.);
    if(vtxChi2 < 1.) {
      rVtxHisto1->Fill(vtxR, 1.);
      if(vtxR > 0.1) {
	step2massHisto->Fill(theKshorts[ksndx].mass(), 1.);
	vtxSigHisto1->Fill(vtxSig, 1.);
	if(vtxSig > 22.) {
	  step3massHisto->Fill(theKshorts[ksndx].mass(), 1.);
	}
      }
    }

    if(vtxChi2 < 1.) {
      rVtxHisto2->Fill(vtxRSph, 1.);
      if(vtxRSph > 0.1) {
	vtxSigHisto2->Fill(vtxSigSph, 1.);
      }
    }

    //theKshorts[ksndx]


    //if(theKshorts.size() < 2) {
    myEtaEfficiencyHisto->Fill(recoEta, 1.);
    myRhoEfficiencyHisto->Fill(recoRad, 1.);
    //}
    
    std::vector<reco::Track> theRefTracks = k0s.refittedTracks();
    
    /*for (unsigned int ndx = 0; ndx < theRefTracks.size(); ndx++) {

      std::cout << "IN LOOP, THIS ISN'T THE PROBLEM. " 
		<< theRefTracks.size() << ": " << std::endl;

      //reco::Track tmpTk( *(k0s.originalTrack( theRefTracks[ndx] )) );
      reco::TrackBaseRef tmpTkRef = k0s.originalTrack( theRefTracks[ndx] );
      std::cout << "Got the TrackBaseRef." << std::endl;
      reco::TrackRef tmpTkRef2 = tmpTkRef.castTo<reco::TrackRef>();
      std::cout << "Created the TrackRef." << std::endl;
      reco::Track tmpTk( *(tmpTkRef2) );
      std::cout << "Did I even get here???" << std::endl;
      myImpactParameterHisto->Fill( sqrt( tmpTk.dxy( beamSpot )
					  * tmpTk.dxy( beamSpot )
					  + tmpTk.dsz( beamSpot )
					  * tmpTk.dsz( beamSpot ) ), 1.);
					  }*/
    

    reco::TransientTrack refTemp1(theRefTracks[0], &(*bFieldHandle));
    reco::TransientTrack refTemp2(theRefTracks[1], &(*bFieldHandle));

    /*reco::TrackBaseRef tkBRef1 = *k0s.tracks_begin();
    reco::TrackBaseRef tkBRef2 = *(++k0s.tracks_begin());//tkBIt2;

    reco::TransientTrack temp1( *( tkBRef1.castTo<reco::TrackRef>() ),
				&( *bFieldHandle ) );
    reco::TransientTrack temp2( *( tkBRef2.castTo<reco::TrackRef>() ),
    &( *bFieldHandle ) );*/
    reco::TransientTrack temp1( *theDaughterTracks[0], &( *bFieldHandle) );
    reco::TransientTrack temp2( *theDaughterTracks[1], &( *bFieldHandle) );

    math::XYZPoint tk1InnerHitPos = temp1.track().innerPosition();
    math::XYZPoint tk2InnerHitPos = temp2.track().innerPosition();
    math::XYZVector difference = tk2InnerHitPos - tk1InnerHitPos;
    double dist = sqrt( difference.x()*difference.x()
			+ difference.y()*difference.y()
			+ difference.z()*difference.z() );
    myInnermostHitDistanceHisto->Fill(dist, 1.);

    TrajectoryStateClosestToBeamLine
      tscb1(temp1.stateAtBeamLine());
    TrajectoryStateClosestToBeamLine
      tscb2(temp2.stateAtBeamLine());
    myImpactParameterHisto2->Fill(tscb1.transverseImpactParameter().value(), 1.);
    myImpactParameterHisto2->Fill(tscb2.transverseImpactParameter().value(), 1.);


    TrajectoryStateClosestToPoint
      tscpRef1(refTemp1.trajectoryStateClosestToPoint(vtxPos));
    TrajectoryStateClosestToPoint
      tscpRef2(refTemp2.trajectoryStateClosestToPoint(vtxPos));
    TrajectoryStateClosestToPoint
      tscp1(temp1.trajectoryStateClosestToPoint(vtxPos));
    TrajectoryStateClosestToPoint
      tscp2(temp2.trajectoryStateClosestToPoint(vtxPos));
    
    GlobalVector refTrack1P(tscpRef1.momentum());
    GlobalVector refTrack2P(tscpRef2.momentum());
    GlobalVector track1P(tscp1.momentum());
    GlobalVector track2P(tscp2.momentum());

    // Find the total momentum of the tracks and its magnitude
    GlobalVector refTotalP(refTrack1P + refTrack2P);
    GlobalVector totalP(track1P + track2P);
    double refPTotMag = sqrt(refTotalP.x()*refTotalP.x()
			     +refTotalP.y()*refTotalP.y()
			     +refTotalP.z()*refTotalP.z());
    double pTotMag = sqrt(totalP.x()*totalP.x() 
			  + totalP.y()*totalP.y()
			  + totalP.z()*totalP.z());

    // Find the total energy of the tracks
    double refETot = sqrt(refTrack1P.x()*refTrack1P.x()
			  + refTrack1P.y()*refTrack1P.y()
			  + refTrack1P.z()*refTrack1P.z()
			  + piMassSq) 
      + sqrt(refTrack2P.x()*refTrack2P.x()
	     + refTrack2P.y()*refTrack2P.y()
	     + refTrack2P.z()*refTrack2P.z()
	     + piMassSq);
    double eTot = sqrt(track1P.x()*track1P.x()
		       + track1P.y()*track1P.y()
		       + track1P.z()*track1P.z()
		       + piMassSq) 
      + sqrt(track2P.x()*track2P.x()
	     + track2P.y()*track2P.y()
	     + track2P.z()*track2P.z()
	     + piMassSq);

    // Calculate the invariant mass
    double refKShortMass = sqrt((refETot+refPTotMag)*(refETot-refPTotMag));
    double kShortMass = sqrt((eTot+pTotMag) * (eTot-pTotMag));

    myRefitKshortMassHisto->Fill(refKShortMass);
    myKshortMassHisto->Fill(kShortMass);
  }
  std::cout << "At end of analyze() function" << std::endl;
}



// ------------ method called once each job just after ending the event loop  ------------
void V0Analyzer::endJob() {

  myEtaEfficiencyHisto->Divide(mySimEtaHisto);
  myRhoEfficiencyHisto->Divide(mySimRhoHisto);
  std::cout << "Pointer address: " << theHistoFile << std::endl;
  theHistoFile->Write();
  delete theHistoFile;
  theHistoFile=0;

}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(V0Analyzer);
