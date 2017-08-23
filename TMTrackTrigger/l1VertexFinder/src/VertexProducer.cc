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
#include "TMTrackTrigger/VertexFinder/interface/VertexFinder.h"
#include "TMTrackTrigger/VertexFinder/interface/L1fittedTrack.h"

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
  clusterTruthInputTag( consumes<TTClusterAssMap>( iConfig.getParameter<edm::InputTag>("clusterTruthInputTag") ) ),
  l1TracksToken_( consumes<TTTrackCollection>(iConfig.getParameter<edm::InputTag>("l1Tracks")) )
{
  // Get configuration parameters
  settings_ = new Settings(iConfig);

  // Tame debug printout.
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(4);

  // Book histograms.
  hists_ = new Histos( settings_ );
  hists_->book();

  //--- Define EDM output to be written to file (if required) 

//  // L1 tracks found by Hough Transform without any track fit.
//  produces< TTTrackCollection >( "TML1TracksHT" ).setBranchAlias("TML1TracksHT");
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

}

void VertexProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // edm::Handle<TrackingParticleCollection> tpHandle;
  // edm::EDGetToken token( consumes<edm::View<TrackingParticleCollection>>( edm::InputTag( "mix", "MergedTrackTruth" ) ) );
  // iEvent.getByToken(inputTag, tpHandle );


  // Note useful info about MC truth particles and about reconstructed stubs .
  InputData inputData(iEvent, iSetup, settings_, tpInputTag, stubInputTag, stubTruthInputTag, clusterTruthInputTag );

  // const vector<TP>&          vTPs   = inputData.getTPs();
  // const vector<const Stub*>& vStubs = inputData.getStubs(); 

  // cout<<"INPUT #TPs = "<<vTPs.size()<<" #STUBs = "<<vStubs.size()<<endl;

  // //=== Fill histograms with stubs and tracking particles from input data.
  // hists_->fillInputData(inputData);


  edm::Handle<TTTrackCollection> l1TracksHandle;
  iEvent.getByToken(l1TracksToken_, l1TracksHandle);

  std::vector<vertexFinder::L1fittedTrack> l1Tracks;
  l1Tracks.reserve(l1TracksHandle->size());
  for(const auto& track : *l1TracksHandle)
    l1Tracks.push_back(vertexFinder::L1fittedTrack(track));

  std::vector<const vertexFinder::L1fittedTrack*> l1TrackPtrs;
  l1TrackPtrs.reserve(l1Tracks.size());
  for(const auto& track : l1Tracks)
    l1TrackPtrs.push_back(&track);



  // FIXME: Check with Davide if the tracks should be filtered using the following cuts
  //   fittedTracks[i].second.accepted() and fittedTracks[i].second.chi2dof()< settings_->chi2OverNdfCut()
  VertexFinder vf(l1TrackPtrs, settings_);
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



  // //=== Store output EDM track and hardware stub collections.
  // iEvent.put( std::move( htTTTracksForOutput ),  "TML1TracksHT");
  // for (const string& fitterName : settings_->trackFitters()) {
  //   string edmName = string("TML1Tracks") + fitterName;
  //   iEvent.put(std::move( allFitTTTracksForOutput[locationInsideArray[fitterName]] ), edmName);
  // }
}


void VertexProducer::endJob() 
{
  hists_->endJobAnalysis();

  cout<<endl<<"Number of (eta,phi) sectors used = (" << settings_->numEtaRegions() << "," << settings_->numPhiSectors()<<")"<<endl; 
}

DEFINE_FWK_MODULE(VertexProducer);
