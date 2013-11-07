// -*- C++ -*-
//
// Package:    SimpleMuon
// Class:      SimpleMuon
// 
/**\class SimpleMuon SimpleMuon.cc GEMCode/SimpleMuon/plugins/SimpleMuon.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Sven Dildick
//         Created:  Wed, 06 Nov 2013 20:05:53 GMT
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "GEMCode/SimMuL1/interface/MatchCSCMuL1.h"
#include "GEMCode/SimMuL1/interface/MuGeometryHelpers.h"
#include "GEMCode/SimMuL1/interface/PSimHitMap.h"

#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

//
// class declaration
//

class SimpleMuon : public edm::EDAnalyzer 
{
public:
  explicit SimpleMuon(const edm::ParameterSet&);
  ~SimpleMuon();
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  
  
private:
  virtual void beginJob() override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() override;
  void propagateToCSCStations(MatchCSCMuL1*);
  TrajectoryStateOnSurface propagateSimTrackToZ(const SimTrack*, const SimVertex*, double);
  void matchSimTrack2SimHits(MatchCSCMuL1*, const edm::SimTrackContainer&, 
			     const edm::SimVertexContainer&, const edm::PSimHitContainer*);

  //virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
  //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  
  // ----------member data ---------------------------
  
  const CSCGeometry* cscGeometry;
  edm::ESHandle<MuonDetLayerGeometry> muonGeometry;

  // propagators
  edm::ESHandle<Propagator> propagatorAlong;
  edm::ESHandle<Propagator> propagatorOpposite;
  edm::ESHandle<MagneticField> theBField;

  bool doStrictSimHitToTrackMatch_;

  int minBxALCT_;
  int maxBxALCT_;
  int minBxCLCT_;
  int maxBxCLCT_;
  int minBxLCT_;
  int maxBxLCT_;
  int minBxMPLCT_;
  int maxBxMPLCT_;
  int minBxGMT_;
  int maxBxGMT_;
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SimpleMuon::SimpleMuon(const edm::ParameterSet& iConfig)
{
  doStrictSimHitToTrackMatch_ = iConfig.getUntrackedParameter<bool>("doStrictSimHitToTrackMatch", false);

  minBxALCT_ = iConfig.getUntrackedParameter< int >("minBxALCT",5);
  maxBxALCT_ = iConfig.getUntrackedParameter< int >("maxBxALCT",7);
  minBxCLCT_ = iConfig.getUntrackedParameter< int >("minBxCLCT",5);
  maxBxCLCT_ = iConfig.getUntrackedParameter< int >("maxBxCLCT",7);
  minBxLCT_ = iConfig.getUntrackedParameter< int >("minBxLCT",5);
  maxBxLCT_ = iConfig.getUntrackedParameter< int >("maxBxLCT",7);
  minBxMPLCT_ = iConfig.getUntrackedParameter< int >("minBxMPLCT",5);
  maxBxMPLCT_ = iConfig.getUntrackedParameter< int >("maxBxMPLCT",7);

}


SimpleMuon::~SimpleMuon()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
SimpleMuon::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // ================================================================================================ 

  //                   G E O M E T R Y   A N D   M A G N E T I C   F I E L D 

  // ================================================================================================ 

  // geometry
  edm::ESHandle<CSCGeometry> cscGeom;
  iSetup.get<MuonGeometryRecord>().get(cscGeom);
  iSetup.get<MuonRecoGeometryRecord>().get(muonGeometry);
  cscGeometry = &*cscGeom;

  // csc trigger geometry
  CSCTriggerGeometry::setGeometry(cscGeometry);

  //Get the Magnetic field from the setup
  iSetup.get<IdealMagneticFieldRecord>().get(theBField);

  // Get the propagators
  iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyRK", propagatorAlong);
  iSetup.get<TrackingComponentsRecord>().get("SmartPropagatorAnyOpposite", propagatorOpposite);

  // ================================================================================================ 

  //                          P A R T I C L E   C O L L E C T I O N S

  // ================================================================================================ 

  // get generator level particle collection
  edm::Handle< reco::GenParticleCollection > hMCCand;
  iEvent.getByLabel("genParticles", hMCCand);
  const reco::GenParticleCollection & cands  = *(hMCCand.product()); 
  
  // get SimTracks
  edm::Handle< edm::SimTrackContainer > hSimTracks;
  iEvent.getByLabel("g4SimHits", hSimTracks);
  const edm::SimTrackContainer & simTracks = *(hSimTracks.product());

  // get simVertices
  edm::Handle< edm::SimVertexContainer > hSimVertices;
  iEvent.getByLabel("g4SimHits", hSimVertices);
  const edm::SimVertexContainer & simVertices = *(hSimVertices.product());

  // ================================================================================================ 

  // 

  // ================================================================================================ 

  const bool debug(true);

  // Select the good generator level muons
  std::vector<const reco::GenParticle *> goodGenMuons;
  for ( size_t ic = 0; ic < cands.size(); ic++ )
  {
    const reco::GenParticle * cand = &(cands[ic]);

    // is this particle a MC muon?    
    if (abs(cand->pdgId()) != 13) continue;
    
    // good MC particle?
    if (cand->status() != 1) continue;
      
    const double mcpt(cand->pt());
    const double mceta(cand->eta());
    const double mcphi(normalizedPhi(cand->phi()));

    // ignore muons with huge eta
    if (fabs(mceta)>10) continue;

    if (debug) std::cout << "Is good MC muon: pt: " << mcpt << ", eta: " << mceta << ", and phi: " << mcphi << std::endl;
    goodGenMuons.push_back(cand);
      
  }
  if (debug) std::cout << "Number of generator level muons " << goodGenMuons.size() << std::endl;

  // get the primary vertex for this simtrack collection
  int no = 1;
  int primaryVert = -1;
  for (edm::SimTrackContainer::const_iterator istrk = simTracks.begin(); istrk != simTracks.end(); ++istrk)
  {
    // print out: simtrack number, simtrack id, particle index, (px, py, pz, E), vertex index, generator level index (-1 if no generator level particle) 
    std::cout<<no<<":\t"<<istrk->trackId()<<" "<<*istrk<<std::endl;
    if ( primaryVert == -1 && !(istrk->noVertex()) )
    {
      primaryVert = istrk->vertIndex();
      std::cout << " -- primary vertex: " << primaryVert << std::endl;
    }
    ++no;
  }
  if ( primaryVert == -1 ) 
  { 
    // No primary vertex found, in non-empty simtrack collection
    std::cout<<">>> WARNING: NO PRIMARY SIMVERTEX! <<<"<<std::endl; 
    if (simTracks.size()>0) return;
  }

  // select the good simulation level muons
  edm::SimTrackContainer goodSimMuons;
  for (edm::SimTrackContainer::const_iterator track = simTracks.begin(); track != simTracks.end(); ++track)
  {
    int i = track - simTracks.begin();

    // sim track is a muon
    if (abs(track->type()) != 13) continue;
    
    const double sim_pt(sqrt(track->momentum().perp2()));
    const double sim_eta(track->momentum().eta());
    const double sim_phi(normalizedPhi(track->momentum().phi()));
    
    // ignore muons with very low pt
    if (sim_pt<2.) continue;

    // track has no primary vertex
    if (!(track->vertIndex() == primaryVert)) continue;
    
    // eta selection - has to be in CSC eta !!!
    if (fabs (sim_eta) > 2.5 || fabs (sim_eta) < .8 ) continue;

    // MC matching of SimMuon to GenMuon
    double mc_eta_match = 999, mc_phi_match = 999;
    if (debug) std::cout << "Sim Muon: " << i << std::endl;

    for (unsigned j=0; j<goodGenMuons.size(); ++j)
    {
      if (debug) std::cout << "   MC Muon: " << j << std::endl;
      auto cand(goodGenMuons.at(i));
      double mc_eta(cand->eta());
      double mc_phi(normalizedPhi(cand->phi()));

      const double dr(deltaR(mc_eta, mc_phi, sim_eta, sim_phi));
      if (debug) std::cout << "   dR = " << dr << std::endl;

      //check if the match makes sense
      if (dr < 0.03 && (mc_eta*sim_eta>0))
      {
	mc_eta_match = mc_eta;
	mc_phi_match = mc_phi;
      }
    }
    // ignore the simtrack if there is no GEN level track
    if (mc_eta_match == 999 && mc_phi_match == 999)
    {
      if (debug) std::cout<<">>> WARNING: no matching MC muon for this sim muon! <<<"<<std::endl;      
      continue;	
    }    
    else
    {
      if (debug) std::cout << ">>> INFO: MC muon was matched to Sim muon" << std::endl;
      if (debug) std::cout << ">>> mc_eta = " << mc_eta_match << ", mc_phi = " << mc_phi_match << ", sim_eta = " << sim_eta << ", sim_phi = " << sim_phi << std::endl; 
    }
    // add the muon to the good sim muons
    goodSimMuons.push_back(*track);

    // store the SIM level and GEN level information
    //    etrk_.mc_pt[nevt]  = (float) mcpt;
    //    etrk_.mc_eta[nevt] = (float) mceta;
    //    etrk_.mc_phi[nevt] = (float) mcphi;
    // 	  etrk_.st_pt[nevt]  = (float) stpt;
    // 	  etrk_.st_eta[nevt] = (float) steta;
    // 	  etrk_.st_phi[nevt] = (float) stphi;
    // 	  etrk_.has_mc_match[nevt] = 1;
    
  }
  if (debug) std::cout << "Number of good simulation level muons " << goodSimMuons.size() << std::endl;


  /*
  // calculate the dR's between all simtracks 
  if (debug) std::cout << "dR between the two good simtracks: "
		       << deltaR(goodSimMuons.at(0).momentum().eta(), normalizedPhi(goodSimMuons.at(0).momentum().phi()),
				 goodSimMuons.at(1).momentum().eta(), normalizedPhi(goodSimMuons.at(1).momentum().phi())) 
		       << std::endl;
  */

  // ================================================================================================ 

  //                   M A I N    L O O P    O V E R   S I M T R A C K S 

  // ================================================================================================ 
  for (edm::SimTrackContainer::const_iterator track = goodSimMuons.begin(); track != goodSimMuons.end(); ++track)
  {
    // create a new matching object for this simtrack 
    MatchCSCMuL1 * match = new MatchCSCMuL1(&*track, &(simVertices[track->vertIndex()]), cscGeometry);
    
    match->muOnly = doStrictSimHitToTrackMatch_;
    match->minBxALCT  = minBxALCT_;
    match->maxBxALCT  = maxBxALCT_;
    match->minBxCLCT  = minBxCLCT_;
    match->maxBxCLCT  = maxBxCLCT_;
    match->minBxLCT   = minBxLCT_;
    match->maxBxLCT   = maxBxLCT_;
    match->minBxMPLCT = minBxMPLCT_;
    match->maxBxMPLCT = maxBxMPLCT_;
    
    // get the approximate position at the CSC stations
    propagateToCSCStations(match);
    
    // match SimHits and do some checks
    matchSimTrack2SimHits(match, simTracks, simVertices, allCSCSimHits);
    
  }



//   double deltaR2Tr = -1;
//   if (sim_n>1) {
//     deltaR2Tr = deltaR(sim_eta[0],sim_phi[0],sim_eta[1],sim_phi[1]);
//     if (deltaR2Tr>M_PI && 1) std::cout<<"PI<deltaR2Tr="<<deltaR2Tr<<std::endl;
    
//     // select only well separated or close simtracks
//     // if (fabs(minSimTrackDR_)>0.01) {
//     //   if (minSimTrackDR_>0. && deltaR2Tr < minSimTrackDR_ ) return true;
//     //   if (minSimTrackDR_<0. && deltaR2Tr > fabs(minSimTrackDR_) ) return true;
//     //}
//   }


      

}


// ------------ method called once each job just before starting event loop  ------------
void 
SimpleMuon::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SimpleMuon::endJob() 
{
}

// ------------ method called when starting to processes a run  ------------
/*
void 
SimpleMuon::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a run  ------------
/*
void 
SimpleMuon::endRun(edm::Run const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when starting to processes a luminosity block  ------------
/*
void 
SimpleMuon::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void 
SimpleMuon::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
SimpleMuon::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}


// ================================================================================================
void 
SimpleMuon::propagateToCSCStations(MatchCSCMuL1 *match)
{
  TrajectoryStateOnSurface tsos;

  // z planes
  const int endcap((match->strk->momentum().eta() >= 0) ? 1 : -1);
  const double zME11(endcap*585.);
  const double zME1(endcap*615.);
  const double zME2(endcap*830.);
  const double zME3(endcap*935.);
  
  // extrapolate to ME1/1 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME11);
  if (tsos.isValid()) match->pME11 = math::XYZVectorD(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());

  // extrapolate to ME1 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME1);
  if (tsos.isValid()) match->pME1 = math::XYZVectorD(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());
     
  // extrapolate to ME2 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME2);
  if (tsos.isValid()) match->pME2 = math::XYZVectorD(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());

  // extrapolate to ME3 surface
  tsos = propagateSimTrackToZ(match->strk, match->svtx, zME3);
  if (tsos.isValid()) match->pME3 = math::XYZVectorD(tsos.globalPosition().x(), tsos.globalPosition().y(), tsos.globalPosition().z());
}

// ================================================================================================
TrajectoryStateOnSurface
SimpleMuon::propagateSimTrackToZ(const SimTrack *track, const SimVertex *vtx, double z)
{
  const Plane::PlanePointer myPlane(Plane::build(Plane::PositionType(0, 0, z), Plane::RotationType()));
  const GlobalPoint  innerPoint(vtx->position().x(),  vtx->position().y(),  vtx->position().z());
  const GlobalVector innerVec  (track->momentum().x(),  track->momentum().y(),  track->momentum().z());
  const FreeTrajectoryState stateStart(innerPoint, innerVec, track->charge(), &*theBField);
  
  TrajectoryStateOnSurface stateProp(propagatorAlong->propagate(stateStart, *myPlane));
  if (!stateProp.isValid()) stateProp = propagatorOpposite->propagate(stateStart, *myPlane);

  return stateProp;
}


// ================================================================================================
void 
SimpleMuon::matchSimTrack2SimHits(MatchCSCMuL1 * match, 
				  const edm::SimTrackContainer & simTracks, 
				  const edm::SimVertexContainer & simVertices, 
				  const edm::PSimHitContainer * cscSimHits)
{
  // Matching of SimHits that were created by SimTrack

  // collect all ID of muon SimTrack children
  match->familyIds = fillSimTrackFamilyIds(match->strk->trackId(), simTracks, simVertices);

  // match SimHits to SimTracks
  std::vector<PSimHit> matchingSimHits = hitsFromSimTrack(match->familyIds, theCSCSimHitMap);
  for (unsigned i=0; i<matchingSimHits.size();i++) {
    if (goodChambersOnly_)
      if ( theStripConditions->isInBadChamber( CSCDetId( matchingSimHits[i].detUnitId() ) ) ) continue; // skip 'bad' chamber
    match->addSimHit(matchingSimHits[i]);
  }

  // checks
  unsigned stNhist = 0;
  for (edm::PSimHitContainer::const_iterator hit = allCSCSimHits->begin();  hit != allCSCSimHits->end();  ++hit) 
    {
      if (hit->trackId() != match->strk->trackId()) continue;
      CSCDetId chId(hit->detUnitId());
      if ( chId.station() == 1 && chId.ring() == 4 && !doME1a_) continue;
      stNhist++;
    }
  if (doStrictSimHitToTrackMatch_ && stNhist != match->simHits.size()) 
    {
      std::cout <<" ALARM!!! matchSimTrack2SimHits: stNhist != stHits.size()  ---> "<<stNhist <<" != "<<match->simHits.size()<<std::endl;
      stNhist = 0;
      if (debugALLEVENT) for (edm::PSimHitContainer::const_iterator hit = allCSCSimHits->begin();  hit != allCSCSimHits->end();  ++hit) 
			   if (hit->trackId() == match->strk->trackId()) {
			     CSCDetId chId(hit->detUnitId());
			     if ( !(chId.station() == 1 && chId.ring() == 4 && !doME1a_) ) std::cout<<"   "<<chId<<"  "<<(*hit)<<" "<<hit->momentumAtEntry()<<" "<<hit->energyLoss()<<" "<<hit->particleType()<<" "<<hit->trackId()<<std::endl;
			   }
    }

  if (debugALLEVENT) {
    std::cout<<"--- SimTrack hits: "<< match->simHits.size()<<std::endl;
    for (unsigned j=0; j<match->simHits.size(); j++) {
      PSimHit & sh = (match->simHits)[j];
      std::cout<<"   "<<sh<<" "<<sh.exitPoint()<<"  "<<sh.momentumAtEntry()<<" "<<sh.energyLoss()<<" "<<sh.particleType()<<" "<<sh.trackId()<<std::endl;
    }
  }
}



//define this as a plug-in
DEFINE_FWK_MODULE(SimpleMuon);
