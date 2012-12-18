#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include "DataFormats/Common/interface/Handle.h"

#include <iostream>
#include <iomanip>

namespace
{
  bool higherPt(const reco::CandidateBaseRef& muon1, const reco::CandidateBaseRef& muon2)
  {
    return (muon1->pt() > muon2->pt());
  }
}

std::vector<reco::CandidateBaseRef> getSelMuons(const edm::Event& evt, const edm::InputTag& srcSelMuons)
{
  std::vector<reco::CandidateBaseRef> selMuons;

  edm::Handle<reco::CompositeCandidateCollection> combCandidatesHandle;
  if ( evt.getByLabel(srcSelMuons, combCandidatesHandle) ) {
    if ( combCandidatesHandle->size() >= 1 ) {
      const reco::CompositeCandidate& combCandidate = combCandidatesHandle->at(0); // TF: use only the first combined candidate
      for ( size_t idx = 0; idx < combCandidate.numberOfDaughters(); ++idx ) { 
	const reco::Candidate* daughter = combCandidate.daughter(idx);
	reco::CandidateBaseRef selMuon;
	if ( daughter->hasMasterClone() ) {
	  selMuon = daughter->masterClone();
	} 
	if ( selMuon.isNull() ) 
	  throw cms::Exception("Configuration") 
	    << "Collection 'selectedMuons' = " << srcSelMuons.label() << " of CompositeCandidates does not refer to daughters of valid type !!\n";
	selMuons.push_back(selMuon);
      }
    }
  } else {
    typedef edm::View<reco::Candidate> CandidateView;
    edm::Handle<CandidateView> selMuonsHandle;
    if ( evt.getByLabel(srcSelMuons, selMuonsHandle) ) {
      for ( size_t idx = 0; idx < selMuonsHandle->size(); ++idx ) {
	selMuons.push_back(reco::CandidateBaseRef(selMuonsHandle->refAt(idx)));
      }
    } else {
      throw cms::Exception("Configuration") 
	<< "Invalid input collection 'selectedMuons' = " << srcSelMuons.label() << " !!\n";
    }
  }

  // sort collection of selected muons by decreasing Pt
  std::sort(selMuons.begin(), selMuons.end(), higherPt);

  return selMuons;
}

reco::CandidateBaseRef getTheMuPlus(const std::vector<reco::CandidateBaseRef>& selMuons)
{
//--- return highest Pt muon of positive charge
//
//    NOTE: function assumes that collection of muons passed as function argument is sorted by decreasing Pt
//         (= as returned by 'getSelMuons' function)
  
  for ( std::vector<reco::CandidateBaseRef>::const_iterator selMuon = selMuons.begin();
	selMuon != selMuons.end(); ++selMuon ) {
    if ( (*selMuon)->charge() > +0.5 ) return (*selMuon);
  }

  // no muon of positive charge found
  return reco::CandidateBaseRef();
}

reco::CandidateBaseRef getTheMuMinus(const std::vector<reco::CandidateBaseRef>& selMuons)
{
//--- return highest Pt muon of negative charge
//
//    NOTE: function assumes that collection of muons passed as function argument is sorted by decreasing Pt
//         (= as returned by 'getSelMuons' function)

  for ( std::vector<reco::CandidateBaseRef>::const_iterator selMuon = selMuons.begin();
	selMuon != selMuons.end(); ++selMuon ) {
    if ( (*selMuon)->charge() < -0.5 ) return (*selMuon);
  }

  // no muon of negative charge found
  return reco::CandidateBaseRef();
}

TrackDetMatchInfo getTrackDetMatchInfo(const edm::Event& evt, const edm::EventSetup& es, 
				       TrackDetectorAssociator& trackAssociator, const TrackAssociatorParameters& trackAssociatorParameters, 
				       const reco::Candidate* muon)
{
  TrackDetMatchInfo trackDetMatchInfo;
  const reco::Muon* recoMuon = dynamic_cast<const reco::Muon*>(muon);
  if ( recoMuon && recoMuon->globalTrack().isNonnull() ) {
    trackDetMatchInfo = trackAssociator.associate(evt, es, *recoMuon->globalTrack(), trackAssociatorParameters);
  } else {
    GlobalVector muonP3(muon->px(), muon->py(), muon->pz()); 
    GlobalPoint muonVtx(muon->vertex().x(), muon->vertex().y(), muon->vertex().z());
    trackDetMatchInfo = trackAssociator.associate(evt, es, muonP3, muonVtx, muon->charge(), trackAssociatorParameters);
  }
  return trackDetMatchInfo;
}

void printMuonDetId(const edm::EventSetup& es, uint32_t rawDetId)
{
  DetId detId(rawDetId);
  std::cout << "detId = " << rawDetId << std::endl;
  const GeomDet* geo = 0;
  if        ( detId.det() == DetId::Muon && detId.subdetId() == MuonSubdetId::CSC ) {
    CSCDetId cscDetId(detId);
    std::cout << " CSC:" 
	      << " endcap = " << cscDetId.endcap() << "," 
	      << " station = " << cscDetId.station() << "," 
	      << " ring = " << cscDetId.ring() << "," 
	      << " chamber = " << cscDetId.chamber() << "," 
	      << " layer = " << cscDetId.layer() << std::endl;
    edm::ESHandle<CSCGeometry> cscGeometry;
    es.get<MuonGeometryRecord>().get(cscGeometry);
    geo = cscGeometry->idToDet(detId);
  } else if ( detId.det() == DetId::Muon && detId.subdetId() == MuonSubdetId::RPC ) {
    RPCDetId rpcDetId(detId);
    std::cout << " RPC:" 
	      << " region = " << rpcDetId.region() << "," 
	      << " ring = " << rpcDetId.ring() << "," 
	      << " station = " << rpcDetId.station() << "," 
	      << " sector = " << rpcDetId.sector() << "," 
	      << " layer = " << rpcDetId.layer() << "," 
	      << " subsector = " << rpcDetId.subsector() << "," 
	      << " roll = " << rpcDetId.roll() << std::endl;
    edm::ESHandle<RPCGeometry> rpcGeometry;
    es.get<MuonGeometryRecord>().get(rpcGeometry);
    geo = rpcGeometry->idToDet(detId);
  } else if ( detId.det() == DetId::Muon && detId.subdetId() == MuonSubdetId::DT  ) {
    DTChamberId dtDetId(detId);
    std::cout << " DT (chamber):" 
	      << " wheel = " << dtDetId.wheel() << "," 
	      << " station = " << dtDetId.station() << "," 
	      << " sector = " << dtDetId.sector() << std::endl;
    edm::ESHandle<DTGeometry> dtGeometry;
    es.get<MuonGeometryRecord>().get(dtGeometry);
    geo = dtGeometry->idToDet(detId);
  } else {
    std::cout << " WARNING: detId refers to unknown detector !!" << std::endl; 
  }
  if ( geo ) {
    std::cout << "(position: eta = " << geo->position().eta() << ", phi = " << geo->position().phi() << ", r = " << geo->position().perp() << ")" << std::endl;
  }
}

bool matchMuonDetId(uint32_t rawDetId1, uint32_t rawDetId2)
{
  DetId detId1(rawDetId1);
  DetId detId2(rawDetId2);
  if        ( detId1.det() == DetId::Muon && detId1.subdetId() == MuonSubdetId::CSC && 
	      detId2.det() == DetId::Muon && detId2.subdetId() == MuonSubdetId::CSC ) {
    CSCDetId cscDetId1(detId1);
    CSCDetId cscDetId2(detId2);
    if ( cscDetId1.endcap()  == cscDetId2.endcap()  &&
	 cscDetId1.station() == cscDetId2.station() &&
	 cscDetId1.ring()    == cscDetId2.ring()    &&
	 cscDetId1.chamber() == cscDetId2.chamber() ) return true;
  } else if ( detId1.det() == DetId::Muon && detId1.subdetId() == MuonSubdetId::RPC &&
	      detId2.det() == DetId::Muon && detId2.subdetId() == MuonSubdetId::RPC ) {
    RPCDetId rpcDetId1(detId1);
    RPCDetId rpcDetId2(detId2);
    if ( rpcDetId1.region()  == rpcDetId2.region()  &&
	 rpcDetId1.ring()    == rpcDetId2.ring()    &&
	 rpcDetId1.station() == rpcDetId2.station() &&
	 rpcDetId1.sector()  == rpcDetId2.sector()  ) return true;
  } else if ( detId1.det() == DetId::Muon && detId1.subdetId() == MuonSubdetId::DT  &&
	      detId2.det() == DetId::Muon && detId2.subdetId() == MuonSubdetId::DT  ) {
    DTChamberId dtDetId1(detId1);
    DTChamberId dtDetId2(detId2);
    if ( dtDetId1.wheel()    == dtDetId2.wheel()    &&
	 dtDetId1.station()  == dtDetId2.station()  &&
	 dtDetId1.sector()   == dtDetId2.sector()   ) return true;
  }
    return false;
}
