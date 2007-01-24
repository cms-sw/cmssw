/** \class MuonTrackAnalyzer
 *  Analyzer of the Muon tracks
 *
 *  $Date: 2006/09/01 14:35:48 $
 *  $Revision: 1.4 $
 *  \author R. Bellan - INFN Torino <riccardo.bellan@cern.ch>
 */

#include "Validation/RecoMuon/src/MuonTrackAnalyzer.h"

// Collaborating Class Header
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "Validation/RecoMuon/src/Histograms.h"
#include "Validation/RecoMuon/src/HTrack.h"

#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

using namespace std;
using namespace edm;

/// Constructor
MuonTrackAnalyzer::MuonTrackAnalyzer(const ParameterSet& pset){
  
  // service parameters
  ParameterSet serviceParameters = pset.getParameter<ParameterSet>("ServiceParameters");
  // the services
  theService = new MuonServiceProxy(serviceParameters);
  
  theMuonTrackLabel = pset.getParameter<InputTag>("MuonTrack");
  theSeedCollectionLabel = pset.getParameter<InputTag>("MuonSeed");

  theRootFileName = pset.getUntrackedParameter<string>("rootFileName");
  
  // Sim or Real
  theDataType = pset.getParameter<InputTag>("DataType");
  if(theDataType.label() != "RealData" && theDataType.label() != "SimData")
    cout<<"Error in Data Type!!"<<endl;

  theCSCSimHitLabel = pset.getParameter<InputTag>("CSCSimHit");
  theDTSimHitLabel = pset.getParameter<InputTag>("DTSimHit");
  theRPCSimHitLabel = pset.getParameter<InputTag>("RPCSimHit");

  theEtaRange = (EtaRange) pset.getParameter<int>("EtaRange");
  
  // number of sim tracks
  numberOfSimTracks=0;
  // number of reco tracks
  numberOfRecTracks=0;
}

/// Destructor
MuonTrackAnalyzer::~MuonTrackAnalyzer(){
  if (theService) delete theService;
}

void MuonTrackAnalyzer::beginJob(const EventSetup& eventSetup){
  // Create the root file
  theFile = new TFile(theRootFileName.c_str(), "RECREATE");

  hSimTracks = new HTrackVariables("SimTracks"); 

  hRecoTracksVTXUpdated = new HTrack("RecoTracks","VTX_Updated"); 
  hRecoTracksVTXUpdatedAndRefitted = new HTrack("RecoTracks","VTX_UpdatedAndRefitted"); 

  hRecoTracksVTX = new HTrack("RecoTracks","VTX"); 
  hRecoTracksInner = new HTrack("RecoTracks","Inner"); 
  hRecoTracksOuter = new HTrack("RecoTracks","Outer"); 

  // General Histos
  hChi2 = new TH1F("chi2","#chi^2",200,0,200);
  hChi2VsEta = new TH2F("chi2VsEta","#chi^2 VS #eta",120,-3.,3.,200,0,200);
  
  hChi2Norm = new TH1F("chi2Norm","Normalized #chi^2",400,0,100);
  hChi2NormVsEta = new TH2F("chi2NormVsEta","Normalized #chi^2 VS #eta",120,-3.,3.,400,0,100);
  
  hHitsPerTrack  = new TH1F("HitsPerTrack","Number of hits per track",55,0,55);
  hHitsPerTrackVsEta  = new TH2F("HitsPerTrackVsEta","Number of hits per track VS #eta",
				 120,-3.,3.,55,0,55);
  
  hDof  = new TH1F("dof","Number of Degree of Freedom",55,0,55);
  hDofVsEta  = new TH2F("dofVsEta","Number of Degree of Freedom VS #eta",120,-3.,3.,55,0,55);

  hChi2Prob = new TH1F("chi2Prob","#chi^2 probability",200,0,1);
  hChi2ProbVsEta = new TH2F("chi2ProbVsEta","#chi^2 probability VS #eta",120,-3.,3.,200,0,1);

  hNumberOfTracks = new TH1F("NumberOfTracks","Number of reconstructed tracks per event",200,0,200);
  hNumberOfTracksVsEta = new TH2F("NumberOfTracksVsEta",
				  "Number of reconstructed tracks per event VS #eta",
				  120,-3.,3.,10,0,10);

  hChargeVsEta = new TH2F("ChargeVsEta","Charge vs #eta gen",120,-3.,3.,4,-2.,2.);
  hChargeVsPt = new TH2F("ChargeVsPt","Charge vs P_{T} gen",250,0,200,4,-2.,2.);
  hPtRecVsPtGen = new TH2F("PtRecVsPtGen","P_{T} rec vs P_{T} gen",250,0,200,250,0,200);

  hDeltaPtVsEta = new TH2F("DeltaPtVsEta","#Delta P_{t} vs #eta gen",120,-3.,3.,500,-250.,250.);
  hDeltaPt_In_Out_VsEta = new TH2F("DeltaPt_In_Out_VsEta_","P^{in}_{t} - P^{out}_{t} vs #eta gen",120,-3.,3.,500,-250.,250.);
  
  theFile->cd();
}

void MuonTrackAnalyzer::endJob(){
  if(theDataType.label() == "SimData"){
    cout << endl << endl << "Number of Sim tracks: " << numberOfSimTracks << endl;
  }
  cout << "Number of Reco tracks: " << numberOfRecTracks << endl;

  // Write the histos to file
  theFile->cd();
  
  hChi2->Write();
  hNumberOfTracks->Write();
  hNumberOfTracksVsEta->Write();
  hChargeVsEta->Write();
  hChargeVsPt->Write();
  hPtRecVsPtGen->Write();
  hChi2Norm->Write();
  hHitsPerTrack->Write();
  hDof->Write();
  hChi2Prob->Write();
  hChi2VsEta->Write();
  hChi2NormVsEta->Write();
  hHitsPerTrackVsEta->Write();
  hDofVsEta->Write(); 
  hChi2ProbVsEta->Write(); 
  hDeltaPtVsEta->Write();  
  hDeltaPt_In_Out_VsEta->Write();

  if(theDataType.label() == "SimData"){
    double eff = hRecoTracksVTX->computeEfficiency(hSimTracks);
    hRecoTracksVTXUpdatedAndRefitted->computeEfficiency(hSimTracks);
    cout<<" *Efficiency* = "<< eff <<"%"<<endl<<endl;;
  }
  
  TDirectory * sim = theFile->mkdir("SimTrack");
  sim->cd();
  
  hSimTracks->Write(); 

  hRecoTracksVTXUpdated->Write(theFile);
  hRecoTracksVTXUpdatedAndRefitted->Write(theFile);
  hRecoTracksVTX->Write(theFile);  
  hRecoTracksInner->Write(theFile); 
  hRecoTracksOuter->Write(theFile);

  theFile->Close();
}


void MuonTrackAnalyzer::analyze(const Event & event, const EventSetup& eventSetup){

  // Update the services
  theService->update(eventSetup);

  MuonPatternRecoDumper debug;

  cout << "Run: " << event.id().run() << " Event: " << event.id().event() << endl;

  // Get the RecTrack collection from the event
  Handle<reco::TrackCollection> staTracks;
  event.getByLabel(theMuonTrackLabel, staTracks);
  
  Handle<SimTrackContainer> simTracks;
  
  if(theDataType.label() == "SimData"){

    if(!checkMuonSimHitPresence(event)) return;
    
    // Get the SimTrack collection from the event
    //    event.getByLabel("g4SimHits",simTracks);
    event.getByLabel(theDataType.instance(),simTracks);
    
    // Loop over the Sim tracks
    SimTrackContainer::const_iterator simTrack;
    cout<<"Simulated tracks: "<<simTracks->size()<<endl;
    
    for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack)
      if (abs((*simTrack).type()) == 13) {

	if( !isInTheAcceptance( (*simTrack).momentum().eta()) ) continue; // FIXME!!
	
	numberOfSimTracks++;
	
	cout<<"Simualted muon:"<<endl;
	cout<<"Sim pT: "<<(*simTrack).momentum().perp()<<endl;
	cout<<"Sim Eta: "<<(*simTrack).momentum().eta()<<endl; // FIXME
	
	hSimTracks->Fill((*simTrack).momentum().mag(), 
			 (*simTrack).momentum().perp(), 
			 (*simTrack).momentum().eta(), 
			 (*simTrack).momentum().phi(), 
			 -(*simTrack).type()/ abs((*simTrack).type()) ); // Double FIXME
	
	hNumberOfTracksVsEta->Fill((*simTrack).momentum().eta(), staTracks->size());	
      }    
    cout << endl; 
  }
  
  if(staTracks->size())
    numberOfRecTracks++;
  
  reco::TrackCollection::const_iterator staTrack;
  
  cout<<"Reconstructed tracks: " << staTracks->size() << endl;
  hNumberOfTracks->Fill(staTracks->size());
    
  // Loop over the Rec tracks
  for (staTrack = staTracks->begin(); staTrack != staTracks->end(); ++staTrack) {

    reco::TransientTrack track(*staTrack,&*theService->magneticField(),theService->trackingGeometry()); 

    cout<<"Analizer: New track, chi2: "<<track.chi2()<<" dof: "<<track.ndof()<<endl;
    hChi2->Fill(track.chi2());
    hDof->Fill(track.ndof());
    hChi2Norm->Fill(track.normalizedChi2());
    hHitsPerTrack->Fill(track.found());

    hChi2Prob->Fill( ChiSquaredProbability(track.chi2(),track.ndof()) );

    cout << "State at VTX: " << endl; 
    TrajectoryStateOnSurface vtxTSOS   = track.impactPointState();
    cout << debug.dumpTSOS(vtxTSOS)<<endl;
    hRecoTracksVTX->Fill(vtxTSOS);


    pair<bool,FreeTrajectoryState> resultOfUpdateAtVtx = updateAtVertex(track);   
    if(resultOfUpdateAtVtx.first){
     
      FreeTrajectoryState vtxFTSUp = resultOfUpdateAtVtx.second;

      cout << "State at VTX (updated): " << endl; 
      cout << debug.dumpFTS(vtxFTSUp)<<endl;

      hRecoTracksVTXUpdated->Fill(vtxFTSUp);
    }


    pair<bool,FreeTrajectoryState> resultOfUpdateAndRefitAtVtx = updateAtVertexAndRefit(track);

    if(resultOfUpdateAndRefitAtVtx.first){
      
      FreeTrajectoryState vtxFTSUpAndRefitted = resultOfUpdateAndRefitAtVtx.second;

      cout << "State at VTX (updated and refitted): " << endl;      
      cout << debug.dumpFTS(vtxFTSUpAndRefitted)<<endl;
      
      hRecoTracksVTXUpdatedAndRefitted->Fill(vtxFTSUpAndRefitted);
    }



    cout << "State at the inner surface: " << endl; 
    TrajectoryStateOnSurface innerTSOS = track.innermostMeasurementState();
    cout << debug.dumpTSOS(innerTSOS)<<endl;
    hRecoTracksInner->Fill(innerTSOS);
    
    cout << "State at the outer surface: " << endl; 
    TrajectoryStateOnSurface outerTSOS = track.outermostMeasurementState();
    cout << debug.dumpTSOS(outerTSOS)<<endl;
    hRecoTracksOuter->Fill(outerTSOS);
    
    // Loop over the RecHits
    trackingRecHit_iterator rhbegin = staTrack->recHitsBegin();
    trackingRecHit_iterator rhend = staTrack->recHitsEnd();
    
    int i=1;

    cout<<"Valid RecHits: "<<staTrack->found()<<" invalid RecHits: "<<staTrack->lost()<<endl;
    for(trackingRecHit_iterator recHit = rhbegin; recHit != rhend; ++recHit){
      if((*recHit)->isValid()){
	const GeomDet* geomDet = theService->trackingGeometry()->idToDet((*recHit)->geographicalId());
	double r = geomDet->surface().position().perp();
	double z = geomDet->toGlobal((*recHit)->localPosition()).z();
	cout<< i++ <<" r: "<< r <<" z: "<<z <<" "<<geomDet->toGlobal((*recHit)->localPosition())
	  // <<" Id: "<<debug.dumpMuonId((*recHit)->geographicalId())
	    <<endl;
      }
    }
    if(theDataType.label() == "SimData" && staTracks->size() ){  

       
      SimTrack simTrack = getSimTrack(vtxTSOS,simTracks);


      hChargeVsEta->Fill(simTrack.momentum().eta(),vtxTSOS.charge());
      hChargeVsPt->Fill(simTrack.momentum().perp(),vtxTSOS.charge());
      hPtRecVsPtGen->Fill(simTrack.momentum().perp(),vtxTSOS.globalMomentum().perp());  

      hChi2VsEta->Fill(simTrack.momentum().eta(),track.chi2());
      hChi2NormVsEta->Fill(simTrack.momentum().eta(),track.normalizedChi2());
      hChi2ProbVsEta->Fill(simTrack.momentum().eta(),ChiSquaredProbability(track.chi2(),track.ndof()));     
      hHitsPerTrackVsEta->Fill(simTrack.momentum().eta(),track.found());
      hDofVsEta->Fill(simTrack.momentum().eta(),track.ndof()); 
      
      hDeltaPtVsEta->Fill(simTrack.momentum().eta(),vtxTSOS.globalMomentum().perp()-simTrack.momentum().perp());
      hDeltaPt_In_Out_VsEta->Fill(simTrack.momentum().eta(),
				  innerTSOS.globalMomentum().perp()-outerTSOS.globalMomentum().perp());


      if(resultOfUpdateAtVtx.first){
	FreeTrajectoryState vtxFTSUp = resultOfUpdateAtVtx.second;
	hRecoTracksVTXUpdated->computeResolutionAndPull(vtxFTSUp,simTrack);
      }
      
      if(resultOfUpdateAndRefitAtVtx.first){
	FreeTrajectoryState vtxFTSUpAndRefitted = resultOfUpdateAndRefitAtVtx.second;
	hRecoTracksVTXUpdatedAndRefitted->computeResolutionAndPull(vtxFTSUpAndRefitted,simTrack);
      }

      hRecoTracksVTX->computeResolutionAndPull(vtxTSOS,simTrack);
      hRecoTracksInner->computeResolutionAndPull(innerTSOS,simTrack);
      hRecoTracksOuter->computeResolutionAndPull(outerTSOS,simTrack);
    }

    
  }
  cout<<"---"<<endl;  
}


SimTrack MuonTrackAnalyzer::getSimTrack(TrajectoryStateOnSurface &tsos,
					Handle<SimTrackContainer> simTracks){
  
  // Loop over the Sim tracks
  SimTrackContainer::const_iterator simTrack;
  
  SimTrack result;
  int mu=0;
  for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack)
    if (abs((*simTrack).type()) == 13) { 
      result = *simTrack;
      ++mu;
    }
  
  if(mu != 1) cout << "WARNING!! more than 1 simulated muon!!" <<endl;
  return result;
}

// SimTrack MuonTrackAnalyzer::getSimTrack(TrajectoryStateOnSurface &tsos,
// 					Handle<SimTrackContainer> simTracks){
  
//   // Loop over the Sim tracks
//   SimTrackContainer::const_iterator simTrack;
  
//   SimTrack result;
//   int mu=0;
//   double deltaR_rif = 9999.;
//   for (simTrack = simTracks->begin(); simTrack != simTracks->end(); ++simTrack)
//     if (abs((*simTrack).type()) == 13) { 

//       double deltaR = 0.;
//       if(tsos.isValid())
// 	deltaR = sqrt( pow(tsos.globalMomentum().eta()-simTrack->momentum().eta(),2)+
// 		       pow(tsos.globalMomentum().phi()-simTrack->momentum().phi(),2));    

//       if(deltaR < deltaR_rif){
// 	deltaR_rif = deltaR;
// 	result = *simTrack;
//       }
//       ++mu;
//     }
//   cout<<"Delta R: "<<deltaR_rif<<endl;
  
//   if(mu != 1) cout << "WARNING!! more than 1 simulated muon!!" <<endl;
//   return result;
// }

bool MuonTrackAnalyzer::isInTheAcceptance(double eta){
  switch(theEtaRange){
  case all:
    return ( abs(eta) <= 4.4 ) ? true : false;
  case barrel:
    return ( abs(eta) < 1.1 ) ? true : false;
  case endcap:
    return ( abs(eta) >= 1.1 && abs(eta) <= 2.4 ) ? true : false;  
  default:
    {cout<<"No correct Eta range selected!! "<<endl; return false;}
  }
}


#include "RecoVertex/VertexTools/interface/LinearizedTrackStateFactory.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexTrackUpdator.h"

pair<bool,FreeTrajectoryState>
MuonTrackAnalyzer::updateAtVertex(const reco::TransientTrack & track) const {

  pair<bool,FreeTrajectoryState> result(false,FreeTrajectoryState());
  
  LinearizedTrackStateFactory linFactory;
  
  GlobalPoint glbPos(0.,0.,0.);
  
  AlgebraicSymMatrix mat(3,0);
  mat[0][0] = (20.e-04)*(20.e-04);
  mat[1][1] = (20.e-04)*(20.e-04);
  mat[2][2] = (5.3)*(5.3);

  GlobalError glbErrPos(mat);
  VertexState vertex(glbPos,glbErrPos);
  
  
  RefCountedLinearizedTrackState linTrackState = linFactory.linearizedTrackState(glbPos,track);
  
  KalmanVertexTrackUpdator vtxUpdator;
  
  pair<RefCountedRefittedTrackState, AlgebraicMatrix> refitted = vtxUpdator.trackRefit(vertex,linTrackState);
  
  if(!refitted.first){
    result.first = true;
    result.second = refitted.first->freeTrajectoryState();
  }
  
  return result;
}

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

pair<bool,FreeTrajectoryState>
MuonTrackAnalyzer::updateAtVertexAndRefit(const reco::TransientTrack & track) const {
  
  pair<bool,FreeTrajectoryState> result(false,FreeTrajectoryState());
  
    GlobalPoint glbPos(0.,0.,0.);
    
    AlgebraicSymMatrix mat(3,0);
    mat[0][0] = (20.e-04)*(20.e-04);
    mat[1][1] = (20.e-04)*(20.e-04);
    mat[2][2] = (5.3)*(5.3);
    GlobalError glbErrPos(mat);

    vector<reco::TransientTrack> singleTrackV(1,track) ;
    KalmanVertexFitter kvf(true);
    CachingVertex tv = kvf.vertex(singleTrackV, glbPos, glbErrPos);
    
    if(!tv.tracks().empty()) {
      result.first = true;
      result.second = tv.tracks().front()->refittedState()->freeTrajectoryState();
    }
    
    return result;
}

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

bool MuonTrackAnalyzer::checkMuonSimHitPresence(const Event & event){

  // Get the SimHit collection from the event
  Handle<PSimHitContainer> dtSimHits;
  event.getByLabel(theDTSimHitLabel.instance(),theDTSimHitLabel.label(), dtSimHits);
  
  Handle<PSimHitContainer> cscSimHits;
  event.getByLabel(theCSCSimHitLabel.instance(),theCSCSimHitLabel.label(), cscSimHits);
  
  Handle<PSimHitContainer> rpcSimHits;
  event.getByLabel(theRPCSimHitLabel.instance(),theRPCSimHitLabel.label(), rpcSimHits);  

  Handle<SimTrackContainer> simTracks;
  event.getByLabel(theDataType.instance(),simTracks);
  
  map<unsigned int, vector<const PSimHit*> > mapOfMuonSimHits;
  
  for(PSimHitContainer::const_iterator simhit = dtSimHits->begin();
      simhit != dtSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13) continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }
  
  for(PSimHitContainer::const_iterator simhit = cscSimHits->begin();
      simhit != cscSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13) continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }
  
  for(PSimHitContainer::const_iterator simhit = rpcSimHits->begin();
      simhit != rpcSimHits->end(); ++simhit) {
    if (abs(simhit->particleType()) != 13) continue;
    mapOfMuonSimHits[simhit->trackId()].push_back(&*simhit);
  }

  bool presence = false;

  for (SimTrackContainer::const_iterator simTrack = simTracks->begin(); 
       simTrack != simTracks->end(); ++simTrack){
    
    if (abs(simTrack->type()) != 13) continue;
    
    map<unsigned int, vector<const PSimHit*> >::const_iterator mapIterator = 
      mapOfMuonSimHits.find(simTrack->trackId());
    
    if (mapIterator != mapOfMuonSimHits.end())
      presence = true;
  }
  
  return presence;
}
