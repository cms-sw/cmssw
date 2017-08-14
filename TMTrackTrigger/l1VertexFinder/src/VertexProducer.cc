#include <TMTrackTrigger/VertexFinder/interface/VertexProducer.h>


#include <TMTrackTrigger/TMTrackFinder/interface/InputData.h>
#include <TMTrackTrigger/TMTrackFinder/interface/Settings.h>
#include <TMTrackTrigger/TMTrackFinder/interface/Histos.h>
#include <TMTrackTrigger/TMTrackFinder/interface/Sector.h>
#include <TMTrackTrigger/TMTrackFinder/interface/HTpair.h>
#include <TMTrackTrigger/TMTrackFinder/interface/KillDupFitTrks.h>
#include <TMTrackTrigger/TMTrackFinder/interface/TrackFitGeneric.h>
#include <TMTrackTrigger/TMTrackFinder/interface/L1fittedTrack.h>
#include <TMTrackTrigger/TMTrackFinder/interface/L1fittedTrk4and5.h>
#include <TMTrackTrigger/TMTrackFinder/interface/ConverterToTTTrack.h>
#include "TMTrackTrigger/TMTrackFinder/interface/HTcell.h"
#include "TMTrackTrigger/TMTrackFinder/interface/MuxHToutputs.h"
#include "TMTrackTrigger/TMTrackFinder/interface/VertexFinder.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "boost/numeric/ublas/matrix.hpp"
#include <iostream>
#include <vector>
#include <set>


using namespace std;
using boost::numeric::ublas::matrix;

VertexProducer::VertexProducer(const edm::ParameterSet& iConfig):
  tpInputTag( consumes<TrackingParticleCollection>( iConfig.getParameter<edm::InputTag>("tpInputTag") ) ),
  stubInputTag( consumes<DetSetVec>( iConfig.getParameter<edm::InputTag>("stubInputTag") ) ),
  stubTruthInputTag( consumes<TTStubAssMap>( iConfig.getParameter<edm::InputTag>("stubTruthInputTag") ) ),
  clusterTruthInputTag( consumes<TTClusterAssMap>( iConfig.getParameter<edm::InputTag>("clusterTruthInputTag") ) )

{
  // Get configuration parameters
  settings_ = new Settings(iConfig);

  // Tame debug printout.
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(4);

  // Book histograms.
  hists_ = new Histos( settings_ );
  hists_->book();

  // Create track fitting algorithm (& internal histograms if it uses them)
  for (const string& fitterName : settings_->trackFitters()) {
    fitterWorkerMap_[ fitterName ] = TrackFitGeneric::create(fitterName, settings_);
    fitterWorkerMap_[ fitterName ]->bookHists(); 
  }

  //--- Define EDM output to be written to file (if required) 

  // L1 tracks found by Hough Transform without any track fit.
  produces< TTTrackCollection >( "TML1TracksHT" ).setBranchAlias("TML1TracksHT");
  // L1 tracks after track fit by each of the fitting algorithms under study
  for (const string& fitterName : settings_->trackFitters()) {
    string edmName = string("TML1Tracks") + fitterName;
    produces< TTTrackCollection >(edmName).setBranchAlias(edmName);
  }
}


void VertexProducer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) 
{
  // Get the B-field and store its value in the Settings class.

  edm::ESHandle<MagneticField> magneticFieldHandle;
  iSetup.get<IdealMagneticFieldRecord>().get(magneticFieldHandle);
  const MagneticField* theMagneticField = magneticFieldHandle.product();
  float bField = theMagneticField->inTesla(GlobalPoint(0,0,0)).z(); // B field in Tesla.
  cout<<endl<<"--- B field = "<<bField<<" Tesla ---"<<endl<<endl;

  settings_->setBfield(bField);

  // Initialize track fitting algorithm at start of run (especially with B-field dependent variables).
  for (const string& fitterName : settings_->trackFitters()) {
    fitterWorkerMap_[ fitterName ]->initRun(); 
  }
}

void VertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // edm::Handle<TrackingParticleCollection> tpHandle;
  // edm::EDGetToken token( consumes<edm::View<TrackingParticleCollection>>( edm::InputTag( "mix", "MergedTrackTruth" ) ) );
  // iEvent.getByToken(inputTag, tpHandle );


  // Note useful info about MC truth particles and about reconstructed stubs .
  InputData inputData(iEvent, iSetup, settings_, tpInputTag, stubInputTag, stubTruthInputTag, clusterTruthInputTag );

  const vector<TP>&          vTPs   = inputData.getTPs();
  const vector<const Stub*>& vStubs = inputData.getStubs(); 

  cout<<"INPUT #TPs = "<<vTPs.size()<<" #STUBs = "<<vStubs.size()<<endl;

  //=== Fill histograms with stubs and tracking particles from input data.
  hists_->fillInputData(inputData);

  // Creates matrix of Sector objects, which decide which stubs are in which (eta,phi) sector
  matrix<Sector>  mSectors(settings_->numPhiSectors(), settings_->numEtaRegions());
  // Create matrix of Hough-Transform arrays, with one-to-one correspondence to sectors.
  matrix<HTpair>  mHtPairs(settings_->numPhiSectors(), settings_->numEtaRegions());

  //=== Initialization
  // Create utility for converting L1 tracks from our private format to official CMSSW EDM format.
  const ConverterToTTTrack converter(settings_);
  // Storage for EDM L1 track collection to be produced from Hough transform output (no fit).
  std::unique_ptr<TTTrackCollection>  htTTTracksForOutput(new TTTrackCollection);
  // Storage for EDM L1 track collection to be produced from fitted tracks (one for each fit algorithm being used).
  // auto_ptr cant be stored in std containers, so use C one, together with map noting which element corresponds to which algorithm.
  const unsigned int nFitAlgs = settings_->trackFitters().size();
  std::unique_ptr<TTTrackCollection> allFitTTTracksForOutput[nFitAlgs];
  std::unique_ptr< FitTrackCollection > allTrackFitTracks[nFitAlgs];

  map<string, unsigned int> locationInsideArray;
  unsigned int ialg = 0;
  for (const string& fitterName : settings_->trackFitters()) {
    std::unique_ptr<TTTrackCollection> fitTTTracksForOutput(new TTTrackCollection);
    allFitTTTracksForOutput[ialg] =  std::move( fitTTTracksForOutput );
    std::unique_ptr<FitTrackCollection> fitTracks(new FitTrackCollection);
    allTrackFitTracks[ialg] = std::move(fitTracks);
    locationInsideArray[fitterName] = ialg++;
  }

  //=== Do tracking in the r-phi Hough transform within each sector.

  unsigned ntracks(0);
  // Fill Hough-Transform arrays with stubs.
  for (unsigned int iPhiSec = 0; iPhiSec < settings_->numPhiSectors(); iPhiSec++) {
    for (unsigned int iEtaReg = 0; iEtaReg < settings_->numEtaRegions(); iEtaReg++) {

      Sector& sector = mSectors(iPhiSec, iEtaReg);
      HTpair& htPair = mHtPairs(iPhiSec, iEtaReg);

      // Initialize constants for this sector.
      sector.init(settings_, iPhiSec, iEtaReg); 
      htPair.init(settings_, iPhiSec, iEtaReg, sector.etaMin(), sector.etaMax(), sector.phiCentre());

      for (const Stub* stub: vStubs) {
	// Digitize stub as would be at input to GP. This doesn't need the octant number, since we assumed an integer number of
	// phi digitisation  bins inside an octant. N.B. This changes the coordinates & bend stored in the stub.
	// The cast allows us to ignore the "const".
	if (settings_->enableDigitize()) (const_cast<Stub*>(stub))->digitizeForGPinput(iPhiSec);

	// Check if stub is inside this sector
        bool inside = sector.inside( stub );

        if (inside) {
	  // Check which eta subsectors within the sector the stub is compatible with (if subsectors being used).
	  const vector<bool> inEtaSubSecs =  sector.insideEtaSubSecs( stub );

	  // Digitize stub if as would be at input to HT, which slightly degrades its coord. & bend resolution, affecting the HT performance.
	  if (settings_->enableDigitize()) (const_cast<Stub*>(stub))->digitizeForHTinput(iPhiSec);

	  // Store stub in Hough transform array for this sector, indicating its compatibility with eta subsectors with sector.
	  htPair.store( stub, inEtaSubSecs );
	}
      }

      // Find tracks in r-phi HT array.
      htPair.end(); // Calls htArrayRphi_.end() -> HTBase::end()
      // std::cout << iPhiSec << " " << iEtaReg << " " << htPair.getRphiHT().numTrackCands2D() << std::endl;
      // if ( htPair.getRphiHT().numTrackCands2D() > 0 ) {
      //   std::cout << "Number of tracks after r-phi HT : " << iPhiSec << " " << iEtaReg << " " << htPair.getRphiHT().numTrackCands2D() << std::endl;
      // }
    }
  }

  if (settings_->muxOutputsHT() && settings_->busySectorKill()) {
    // Multiplex outputs of several HT onto one pair of output opto-links.
    // This only affects tracking performance if option busySectorKill is enabled, so that tracks that
    // can't be sent down the link within the time-multiplexed period are killed.
    MuxHToutputs muxHT(settings_);
    muxHT.exec(mHtPairs);
  }

  //=== Optionally run r-z filters or r-z HT. Make 3D tracks.

  for (unsigned int iPhiSec = 0; iPhiSec < settings_->numPhiSectors(); iPhiSec++) {
    for (unsigned int iEtaReg = 0; iEtaReg < settings_->numEtaRegions(); iEtaReg++) {

      HTpair& htPair = mHtPairs(iPhiSec, iEtaReg);
      // std::cout << iPhiSec << " " << iEtaReg << " " << htPair.getRphiHT().numTrackCands2D() << std::endl;
      // Convert these to 3D tracks (optionally by running r-z filters etc.)
      htPair.make3Dtracks();

      // Convert these tracks to EDM format for output (not used by Histos class).
      const vector<L1track3D>& vecTrk3D = htPair.trackCands3D();
      ntracks += vecTrk3D.size();
      for (const L1track3D& trk : vecTrk3D) {
        TTTrack< Ref_Phase2TrackerDigi_ > htTTTrack = converter.makeTTTrack(trk, iPhiSec, iEtaReg);
        htTTTracksForOutput->push_back( htTTTrack );
      }
    }
  }

  // Initialize the duplicate track removal algorithm that can optionally be run after the track fit.
  KillDupFitTrks killDupFitTrks;
  killDupFitTrks.init(settings_, settings_->dupTrkAlgFit());
  
  //=== Do a helix fit to all the track candidates.
  FitTrackCollection fitTracks;
  vector<std::pair<std::string, L1fittedTrack>> fittedTracks;
  for (unsigned int iPhiSec = 0; iPhiSec < settings_->numPhiSectors(); iPhiSec++) {
    for (unsigned int iEtaReg = 0; iEtaReg < settings_->numEtaRegions(); iEtaReg++) {

      HTpair& htPair = mHtPairs(iPhiSec, iEtaReg);

      // In principal, should digitize stubs on track here using digitization relative to this phi sector.
      // However, previously digitized stubs will still be valid if digitization bins in phi align with
      // phi sector boundaries, so couldn't be bothered. If they do not align, the average effect of 
      // digitization on the track fit will be correct, but the effect on individual tracks will not.

      // Get track candidates found by Hough transform in this sector.
      const vector<L1track3D>& vecTrk3D = htPair.trackCands3D();
      // Loop over all the fitting algorithms we are trying.
      for (const string& fitterName : settings_->trackFitters()) {
        // Fit all tracks in this sector
	vector<L1fittedTrack> fittedTracksInSec;
        for (const L1track3D& trk : vecTrk3D) {
	  L1fittedTrack fitTrack = fitterWorkerMap_[fitterName]->fit(trk, iPhiSec, iEtaReg);
	  // Store fitted tracks, such that there is one fittedTracks corresponding to each HT tracks.
	  // N.B. Tracks rejected by the fit are also stored, but marked.
	  fittedTracksInSec.push_back(fitTrack);
	}

	// Run duplicate track removal on the fitted tracks if requested.
	const vector<L1fittedTrack> filtFittedTracksInSec = killDupFitTrks.filter( fittedTracksInSec );

	// Store fitted tracks from entire tracker.
	for (const L1fittedTrack& fitTrk : filtFittedTracksInSec) {
	  fittedTracks.push_back(std::make_pair(fitterName, fitTrk));
	  // Convert these fitted tracks to EDM format for output (not used by Histos class).
	  // Only do this for valid fitted tracks, meaning that these EDM tracks do not correspond 1 to 1 with fittedTracks.
	  if (fitTrk.accepted()) {
	    TTTrack< Ref_Phase2TrackerDigi_ > fitTTTrack = converter.makeTTTrack(fitTrk, iPhiSec, iEtaReg);
	    allFitTTTracksForOutput[locationInsideArray[fitterName]]->push_back(fitTTTrack);
	  }
	}
      }
    }
  }

  // Histogram the undigitized stubs, since with some firwmare versions, quantities like digitized stub dphi are
  // not available, so would give errors when people try histogramming them.
  for (const Stub* stub: vStubs) {
    if (settings_->enableDigitize()) (const_cast<Stub*>(stub))->reset_digitize();
  }

  //=== Fill histograms that check if choice of (eta,phi) sectors is good.
  hists_->fillEtaPhiSectors(inputData, mSectors);

  //=== Fill histograms that look at filling of r-phi HT arrays.
  hists_->fillRphiHT(mHtPairs);

  //=== Fill histograms that look at r-z filters (or other filters run after r-phi HT).
  hists_->fillRZfilters(mHtPairs);

  //=== Fill histograms studying track candidates found by r-phi Hough Transform.
  hists_->fillTrackCands(inputData, mSectors, mHtPairs);

  //=== Fill histograms studying track fitting performance
  hists_->fillTrackFitting(inputData, fittedTracks,  settings_->chi2OverNdfCut() );


  for (const string& fitterName : settings_->trackFitters()) {
    for(unsigned int i = 0; i<fittedTracks.size(); ++i){
      if(fittedTracks[i].first == fitterName and fittedTracks[i].second.accepted() and fittedTracks[i].second.chi2dof()< settings_->chi2OverNdfCut() ) {
        L1fittedTrack* pTrack;
        pTrack = &(fittedTracks[i].second);
        fitTracks.push_back(pTrack);
      }
    }
    
    VertexFinder vf(fitTracks, settings_);
    if(settings_->vx_algoId() == 0){
      cout << "Finding vertices using a gap clustering algorithm "<< endl;
      vf.GapClustering();
    } else if(settings_->vx_algoId() == 1){
      cout << "Finding vertices using a Simple Merge Clustering algorithm "<< endl;
      vf.SimpleMergeClustering();
    } else if(settings_->vx_algoId() == 2){
      cout << "Finding vertices using a DBSCAN algorithm "<< endl;
      vf.DBSCAN();
    } else if(settings_->vx_algoId() == 3){
      cout << "Finding vertices using a PVR algorithm "<< endl;
      vf.PVR();
    } else if(settings_->vx_algoId() == 4){
      cout << "Finding vertices using an AdaptiveVertexReconstruction algorithm "<< endl;
      vf.AdaptiveVertexReconstruction();
    } else if(settings_->vx_algoId() == 5){
      cout << "Finding vertices using an Highest Pt Vertex algorithm "<< endl;
      vf.HPV();
    }
    else{
      cout << "No valid vertex reconstruction algorithm has been selected. Running a gap clustering algorithm "<< endl;
      vf.GapClustering();
    }

    vf.TDRalgorithm();
    vf.FindPrimaryVertex();

    if(settings_->debug()==7 and vf.numVertices() > 0){
      cout << "Num Found Vertices " << vf.numVertices() << endl;
      cout << "Reconstructed Primary Vertex z0 "<<vf.PrimaryVertex().z0() << " pT "<< vf.PrimaryVertex().pT() << endl;
    }
    //=== Fill histograms studying vertex reconstruction performance
    hists_->fillVertexReconstruction(inputData, vf);    
  }



  //=== Store output EDM track and hardware stub collections.
  iEvent.put( std::move( htTTTracksForOutput ),  "TML1TracksHT");
  for (const string& fitterName : settings_->trackFitters()) {
    string edmName = string("TML1Tracks") + fitterName;
    iEvent.put(std::move( allFitTTTracksForOutput[locationInsideArray[fitterName]] ), edmName);
  }
}


void VertexProducer::endJob() 
{
  hists_->endJobAnalysis();

  for (const string& fitterName : settings_->trackFitters()) {
    //cout << "# of duplicated stubs = " << fitterWorkerMap_[fitterName]->nDupStubs() << endl;
    delete fitterWorkerMap_[ string(fitterName) ];
  }

  cout<<endl<<"Number of (eta,phi) sectors used = (" << settings_->numEtaRegions() << "," << settings_->numPhiSectors()<<")"<<endl; 
}

DEFINE_FWK_MODULE(VertexProducer);
