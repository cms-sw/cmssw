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

#include <TGraph.h>

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

void repairBarcodes(HepMC::GenEvent* genEvt)
{
  // AB: Note that we cannot do the barcode re-assignment "inline" without first
  //     creating a copy of the vertex and particle collections, because otherwise
  //     changing a barcode might invalidate the iterator which is used for
  //     iterating over the collection.
  const std::vector<HepMC::GenVertex*> vertices(genEvt->vertices_begin(), genEvt->vertices_end());
  const std::vector<HepMC::GenParticle*> particles(genEvt->particles_begin(), genEvt->particles_end());

  int next_genVtx_barcode = 1;
  for(std::vector<HepMC::GenVertex*>::const_iterator iter = vertices.begin(); iter != vertices.end(); ++iter, ++next_genVtx_barcode)
    while(!(*iter)->suggest_barcode(-next_genVtx_barcode))
      ++next_genVtx_barcode;

  int next_genParticle_barcode = 1;
  for(std::vector<HepMC::GenParticle*>::const_iterator iter = particles.begin(); iter != particles.end(); ++iter, ++next_genParticle_barcode)
    while(!(*iter)->suggest_barcode(next_genParticle_barcode))
      ++next_genParticle_barcode;
}

const reco::GenParticle* findGenParticleForMCEmbedding(const reco::Candidate::LorentzVector& direction,
						       const reco::GenParticleCollection& genParticles, double dRmax, int status,
						       const std::vector<int>* pdgIds, bool pdgIdStrict)
{
  //---------------------------------------------------------------------------
  // NOTE: this function has been copied from TauAnalysis/CandidateTools/src/candidateAuxFunctions.cc
  //       in order to avoid a package dependency of TauAnalysis/MCEmbedding on TauAnalysis/CandidateTools
  //---------------------------------------------------------------------------
  
  bool bestMatchMatchesPdgId = false;
  const reco::GenParticle* bestMatch = 0;
  
  for ( reco::GenParticleCollection::const_iterator genParticle = genParticles.begin();
	genParticle != genParticles.end(); ++genParticle ) {
    bool matchesPdgId = false;
    if ( pdgIds ) {
      for ( std::vector<int>::const_iterator pdgId = pdgIds->begin(); pdgId != pdgIds->end(); ++pdgId ) {
	if ( genParticle->pdgId() == (*pdgId) ) {
	  matchesPdgId = true;
	  break;
	}
      }
    }
    
    // If we require strict PDG id checking, skip it if it doesn't match
    if ( pdgIds && !matchesPdgId && pdgIdStrict ) continue;
    
    // Check if status matches - if not, skip it.
    bool statusMatch = (status == -1 || genParticle->status() == status);
    if ( !statusMatch ) continue;
    
    double dR = reco::deltaR(direction, genParticle->p4());
    if ( dR > dRmax ) continue;
    
    // Check if higher than current best match
    bool higherEnergyThanBestMatch = (bestMatch) ? 
      (genParticle->energy() > bestMatch->energy()) : true;
    
    // Check if old bestMatch was not a prefered ID and the new one is.
    if ( bestMatchMatchesPdgId ) {
      // If the old one matches, only use the new one if it is a better
      // energy match
      if ( matchesPdgId && higherEnergyThanBestMatch ) bestMatch = &(*genParticle);
    } else {
      // The old one doesn't match the prefferred list if it is either
      // a better energy match or better pdgId match
      if ( higherEnergyThanBestMatch || matchesPdgId ) {
	bestMatch = &(*genParticle);
	if ( matchesPdgId ) bestMatchMatchesPdgId = true;
      }
    }
  }
  
  return bestMatch;
}

void compGenParticleP4afterRad(const reco::GenParticle* mother, reco::Candidate::LorentzVector& particleP4_afterRad, int absPdgId)
{
  unsigned numDaughters = mother->numberOfDaughters();
  for ( unsigned iDaughter = 0; iDaughter < numDaughters; ++iDaughter ) {
    const reco::GenParticle* daughter = mother->daughterRef(iDaughter).get();
    
    compGenParticleP4afterRad(daughter, particleP4_afterRad, absPdgId);
  }
  
  if ( abs(mother->pdgId()) == absPdgId ) {
    if ( mother->energy() < particleP4_afterRad.energy() ) particleP4_afterRad = mother->p4();
  }
}

void compGenMuonP4afterRad(const reco::GenParticle* mother, reco::Candidate::LorentzVector& muonP4_afterRad)
{
  return compGenParticleP4afterRad(mother, muonP4_afterRad, 13);
}

void compGenTauP4afterRad(const reco::GenParticle* mother, reco::Candidate::LorentzVector& tauP4_afterRad)
{
  return compGenParticleP4afterRad(mother, tauP4_afterRad, 15);
}

void findMuons(const edm::Event& evt, const edm::InputTag& src, 
	       reco::Candidate::LorentzVector& genMuonPlusP4, bool& genMuonPlus_found, 
	       reco::Candidate::LorentzVector& genMuonMinusP4, bool& genMuonMinus_found)
{
  //std::cout << "<findMuons>:" << std::endl;
  //std::cout << " src = " << src << std::endl;
  typedef std::vector<reco::Particle> ParticleCollection;
  edm::Handle<ParticleCollection> muons;
  evt.getByLabel(src, muons);
  int idx = 0;
  for ( ParticleCollection::const_iterator muon = muons->begin();
	muon != muons->end(); ++muon ) {
    //std::cout << "muon #" << idx << ": Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() << std::endl;
    if ( muon->charge() > +0.5 ) {
      genMuonPlusP4 = muon->p4();
      genMuonPlus_found = true;
    }
    if ( muon->charge() < -0.5 ) {
      genMuonMinusP4 = muon->p4();
      genMuonMinus_found = true;
    }
    ++idx;
  }
}

double getDeDxForPbWO4(double p)
{
  static TGraph* dedxGraphPbWO4 = NULL;

  static const double E[] = { 1.0, 1.40, 2.0, 3.0, 4.0, 8.0, 10.0,
                            14.0, 20.0, 30.0, 40.0, 80.0, 100.0,
                            140.0, 169.0, 200.0, 300.0, 400.0, 800.0, 1000.0,
                            1400.0, 2000.0, 3000.0, 4000.0, 8000.0 };
  static const double DEDX[] = { 1.385, 1.440, 1.500, 1.569, 1.618, 1.743, 1.788,
                               1.862, 1.957, 2.101, 2.239, 2.778, 3.052,
                               3.603, 4.018, 4.456, 5.876, 7.333, 13.283, 16.320,
                               22.382, 31.625, 47.007, 62.559, 125.149 }; // in units of MeV
  static const unsigned int N_ENTRIES = sizeof(E)/sizeof(E[0]);

  if ( !dedxGraphPbWO4 ) dedxGraphPbWO4 = new TGraph(N_ENTRIES, E, DEDX);

  return dedxGraphPbWO4->Eval(p)*1.e-3; // convert to GeV
}

