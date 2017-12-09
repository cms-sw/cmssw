
#include "TMTrackTrigger/l1VertexFinder/interface/InputData.h"


#include <map>

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

// #include "Geometry/Records/interface/StackedTrackerGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

// TTStubAssociationMap.h forgets to two needed files, so must include them here ...
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"

#include "TMTrackTrigger/l1VertexFinder/interface/Settings.h"



using namespace std;
 
namespace l1tVertexFinder {

InputData::InputData(const edm::Event& iEvent, const edm::EventSetup& iSetup, Settings* settings, 
  const edm::EDGetTokenT<TrackingParticleCollection> tpInputTag,
  const edm::EDGetTokenT<DetSetVec> stubInputTag,
  const edm::EDGetTokenT<TTStubAssMap> stubTruthInputTag,
  const edm::EDGetTokenT<TTClusterAssMap> clusterTruthInputTag
   )
  {

  vTPs_.reserve(2500);
// FIXME - filling vStubs_ removed during migration to VertexFinder package; not used in migrated code.
//  vStubs_.reserve(35000);
  vAllStubs_.reserve(35000);

  // Get TrackingParticle info

  edm::Handle<TrackingParticleCollection> tpHandle;
  iEvent.getByToken(tpInputTag, tpHandle );

  unsigned int tpCount = 0;
  float metY = 0;
  float metX = 0;

  float metY_pu = 0;
  float metX_pu = 0;

  genPt_ = 0.;
  genPt_PU_ = 0.; 

  for (unsigned int i = 0; i < tpHandle->size(); i++) {
    TrackingParticlePtr tpPtr(tpHandle, i);
    // Store the TrackingParticle info, using class TP to provide easy access to the most useful info.
    TP tp(tpPtr, tpCount, settings);
    
    if(tp.physicsCollision()){
      metX += tp.pt()*cos(tp.phi0());
      metY += tp.pt()*sin(tp.phi0());
      genPt_ += tp.pt();
    } else{
      metX_pu += tp.pt()*cos(tp.phi0());
      metY_pu += tp.pt()*sin(tp.phi0());
      genPt_PU_ += tp.pt();
    }
    
    // Only bother storing tp if it could be useful for tracking efficiency or fake rate measurements.
    if (tp.use()) {
      vTPs_.push_back( tp );
      // cout << "tracking particle z0 "<< tp.z0() << " vx "<< tp.vx() << " vy "<< tp.vy() << " vz "<< tp.vz() << " physicsCollision " << tp.physicsCollision() << endl;
      tpCount++;
    }
  }

  // Total Generated MET in the event
  genMET_ = sqrt(metX*metX+metY*metY);
  // Total GenMET in PU events
  genMET_PU_ = sqrt(metX*metX+metY*metY);
  cout << "genPt in the event "<< genPt_ << endl;
  cout << "genMET in the event = "<< genMET_ << endl;


  // Also create map relating edm::Ptr<TrackingParticle> to TP.

  map<edm::Ptr< TrackingParticle >, const TP* > translateTP;

  for (const TP& tp : vTPs_) {
    TrackingParticlePtr tpPtr(tp);
    translateTP[tpPtr] = &tp;
  }

  // Get the tracker geometry info needed to unpack the stub info.

  edm::ESHandle<TrackerGeometry> trackerGeometryHandle;
  iSetup.get<TrackerDigiGeometryRecord>().get( trackerGeometryHandle );

  const TrackerGeometry*  trackerGeometry = trackerGeometryHandle.product();

  edm::ESHandle<TrackerTopology> trackerTopologyHandle;
  iSetup.get<TrackerTopologyRcd>().get(trackerTopologyHandle);

  const TrackerTopology*  trackerTopology = trackerTopologyHandle.product();

  // Get stub info, by looping over modules and then stubs inside each module.
  // Also get the association map from stubs to tracking particles.

  edm::Handle<DetSetVec>       ttStubHandle;
  edm::Handle<TTStubAssMap>    mcTruthTTStubHandle;
  edm::Handle<TTClusterAssMap> mcTruthTTClusterHandle;
  // iEvent.getByLabel("TTStubsFromPixelDigis"            , "StubAccepted"    , ttStubHandle           );
  // iEvent.getByLabel("TTStubAssociatorFromPixelDigis"   , "StubAccepted"    , mcTruthTTStubHandle    );
  // iEvent.getByLabel("TTClusterAssociatorFromPixelDigis", "ClusterAccepted" , mcTruthTTClusterHandle );

  iEvent.getByToken(stubInputTag, ttStubHandle );
  iEvent.getByToken(stubTruthInputTag, mcTruthTTStubHandle );
  iEvent.getByToken(clusterTruthInputTag, mcTruthTTClusterHandle );


  std::set<DetId> lStubDetIds;
  for (DetSetVec::const_iterator p_module = ttStubHandle->begin(); p_module != ttStubHandle->end(); p_module++) {
    for (DetSet::const_iterator p_ttstub = p_module->begin(); p_ttstub != p_module->end(); p_ttstub++) {
      lStubDetIds.insert(p_ttstub->getDetId());
    }
  }

  for (auto gd=trackerGeometry->dets().begin(); gd != trackerGeometry->dets().end(); gd++)
  {
    DetId detid = (*gd)->geographicalId();
    if(detid.subdetId()!=StripSubdetector::TOB && detid.subdetId()!=StripSubdetector::TID )
      continue; // only run on OT
    if(!trackerTopology->isLower(detid) )
      continue; // loop on the stacks: choose the lower arbitrarily
    DetId stackDetid = trackerTopology->stack(detid); // Stub module detid

    if ( lStubDetIds.count(stackDetid) > 0 ) {
      assert (stubGeoDetIdMap_.count(stackDetid) == 0);
      stubGeoDetIdMap_[stackDetid] = detid;
    }
  }
  assert (lStubDetIds.size() == stubGeoDetIdMap_.size());

  unsigned int stubCount = 0;
  for (DetSetVec::const_iterator p_module = ttStubHandle->begin(); p_module != ttStubHandle->end(); p_module++) {
    for (DetSet::const_iterator p_ttstub = p_module->begin(); p_ttstub != p_module->end(); p_ttstub++) {
      TTStubRef ttStubRef = edmNew::makeRefTo(ttStubHandle, p_ttstub );
      // Store the Stub info, using class Stub to provide easy access to the most useful info.
      Stub stub(ttStubRef, stubCount, settings, trackerGeometry, trackerTopology, stubGeoDetIdMap_ );
      // Also fill truth associating stubs to tracking particles.
      //      stub.fillTruth(vTPs_, mcTruthTTStubHandle, mcTruthTTClusterHandle); 
      stub.fillTruth(translateTP, mcTruthTTStubHandle, mcTruthTTClusterHandle);
      vAllStubs_.push_back( stub );
      stubCount++;
    }
  }

  std::map<const TP*, std::vector<const Stub*> > tpStubMap;
  for (const TP& tp : vTPs_)
    tpStubMap[&tp] = std::vector<const Stub*>();
  for (const Stub& stub : vAllStubs_) {
    for (const TP* tp : stub.assocTPs()) {
      tpStubMap[tp].push_back(&stub);
    }
  }
  // std::cout << "Number of stubs read in : " << stubCount << std::endl;

// FIXME - filling vStubs_ removed during migration to VertexFinder package; not used in migrated code.
//  // Produced reduced list containing only the subset of stubs that the user has declared will be 
//  // output by the front-end readout electronics.
//  for (const Stub& s : vAllStubs_) {
//    if (s.frontendPass()) vStubs_.push_back( &s );
//  }
  // std::cout << "Number of stubs from FE : " << vStubs_.size() << std::endl;

// FIXME - stub ordering removed during migration to VertexFinder package; should not be required.
//  // Optionally sort stubs according to bend, so highest Pt ones are sent from DTC to GP first.
//  if (settings->orderStubsByBend()) std::sort(vStubs_.begin(), vStubs_.end(), SortStubsInBend());
//  // Note list of stubs produced by each tracking particle.

  // (By passing vAllStubs_ here instead of vStubs_, it means that any algorithmic efficiencies
  // measured will be reduced if the tightened frontend electronics cuts, specified in section StubCuts
  // of Analyze_Defaults_cfi.py, are not 100% efficient).
  for (unsigned int j = 0; j < vTPs_.size(); j++) {
    assert (tpStubMap.count(&vTPs_.at(j)) == 1);
    vTPs_[j].setMatchingStubs(tpStubMap.find(&vTPs_.at(j))->second);
    if(vTPs_[j].useForAlgEff()) {
      
      vertex_.insert(vTPs_[j]);
      // cout << "here "<< endl;
      // bool found = false;
      // unsigned int index = 9999;
      // for(unsigned int i = 0; i < vVertices_.size(); i++){
      //   if(vTPs_[j].vz() == vVertices_[i].z()){
      //     found = true;
      //     index = i;
      //     break;
      //   }
      // }

      // if(found){
      //   vVertices_[index].insert(vTPs_[j]);
      // } else{
      //   Vertex vx(vTPs_[j].vz());
      //   vx.insert(vTPs_[j]);
      //   vVertices_.push_back(vx);
      // }
    } else if(vTPs_[j].useForVertexReco()){
      bool found = false;
      for(unsigned int i = 0 ; i < vertices_.size(); ++i){
        if(vTPs_[j].vz() == vertices_[i].vz()){
          vertices_[i].insert(vTPs_[j]);
          found = true;
          break;
        } 
      }
      if(!found){
        Vertex vertex(vTPs_[j].vz());
        vertex.insert(vTPs_[j]);
        vertices_.push_back(vertex);        
      }
    }
  }

  for(Vertex vertex : vertices_){
    if(vertex.numTracks() >= settings->vx_minTracks()) recoVertices_.push_back(vertex);
  }

  cout << vertices_.size() << " pileup vertices in the event, "<< recoVertices_.size() << " reconstructable" << endl;
  

  vertex_.computeParameters();
  if(settings->debug() == 7) {
    cout << "Vertex "<< vertex_.z0() << " containing "<< vertex_.numTracks() << " total pT "<< vertex_.pT() << endl;
  
  }

  for(unsigned int i = 0; i<vertices_.size(); ++i){
    vertices_[i].computeParameters();
  }

  for(unsigned int i = 0; i<recoVertices_.size(); ++i){
    recoVertices_[i].computeParameters();
  }

  std::sort(vertices_.begin(), vertices_.end(), SortVertexByZ0());
  std::sort(recoVertices_.begin(), recoVertices_.end(), SortVertexByZ0());
}

} // end namespace l1tVertexFinder
