
/** \class MuonPFCandidateCleaner
 *
 * Produce collection of PFCandidates in Z --> mu+ mu- event
 * from which the two muons are removed 
 * (later to be replaced by simulated tau decay products) 
 * 
 * \authors Tomasz Maciej Frueboes;
 *          Christian Veelken, LLR
 *
 * \version $Revision: 1.1 $
 *
 * $Id: MuonPFCandidateCleaner.cc,v 1.1 2013/02/05 19:59:05 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <vector>
#include <algorithm>

class MuonPFCandidateCleaner : public edm::EDProducer 
{
 public:
  explicit MuonPFCandidateCleaner(const edm::ParameterSet&);
  ~MuonPFCandidateCleaner() {}

 private:
  virtual void produce(edm::Event&, const edm::EventSetup&);

  edm::InputTag srcSelectedMuons_;
  edm::InputTag srcPFCandidates_;

  double dRmatch_;
  bool removeDuplicates_;

  int maxWarnings_tooMany_;
  int numWarnings_tooMany_;
  int maxWarnings_tooFew_;
  int numWarnings_tooFew_;

  int verbosity_;
};

MuonPFCandidateCleaner::MuonPFCandidateCleaner(const edm::ParameterSet& cfg)
  : srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons")),
    srcPFCandidates_(cfg.getParameter<edm::InputTag>("pfCands")),
    dRmatch_(cfg.getParameter<double>("dRmatch")),
    removeDuplicates_(cfg.getParameter<bool>("removeDuplicates")),
    maxWarnings_tooMany_(100),
    numWarnings_tooMany_(0),
    maxWarnings_tooFew_(3),
    numWarnings_tooFew_(0)
{
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;

  produces<reco::PFCandidateCollection>();
}

namespace
{
  struct muonToPFCandMatchInfoType
  {
    muonToPFCandMatchInfoType(const reco::Particle::LorentzVector& muonP4, const reco::PFCandidate* pfCandidate, double dR)
      : muonPt_(muonP4.pt()),
	pfCandidatePt_(pfCandidate->pt()),
	pfCandidateCharge_(pfCandidate->charge()),
	dR_(dR),
	pfCandidate_(pfCandidate)
    {}
    ~muonToPFCandMatchInfoType() {}
    double muonPt_;
    double pfCandidatePt_;    
    int pfCandidateCharge_;
    double dR_;
    const reco::PFCandidate* pfCandidate_;
  };

  struct SortMuonToPFCandMatchInfosDescendingMatchQuality
  {
    bool operator() (const muonToPFCandMatchInfoType& m1, const muonToPFCandMatchInfoType& m2)
    {
      // 1st criterion: prefer matches of high Pt
      if ( m1.pfCandidatePt_ > (0.5*m1.muonPt_) && m2.pfCandidatePt_ < (0.5*m2.muonPt_) ) return true;  // m1 has higher rank than m2
      if ( m1.pfCandidatePt_ < (0.5*m1.muonPt_) && m2.pfCandidatePt_ > (0.5*m2.muonPt_) ) return false; // m2 has higher rank than m1
      // 2nd criterion: prefer matches to charged particles
      if ( m1.pfCandidateCharge_ != 0 && m2.pfCandidateCharge_ == 0 ) return true;
      if ( m1.pfCandidateCharge_ == 0 && m2.pfCandidateCharge_ != 0 ) return false;
      // 3rd criterion: in case multiple matches to high Pt, charged particles exist, 
      //                take particle matched most closely in dR
      return (m1.dR_ < m2.dR_); 
    }
  };

  std::string runLumiEventNumbers_to_string(const edm::Event& evt)
  {
    edm::RunNumber_t run_number = evt.id().run();
    edm::LuminosityBlockNumber_t ls_number = evt.luminosityBlock();
    edm::EventNumber_t event_number = evt.id().event();
    std::ostringstream retVal;
    retVal << "Run = " << run_number << ", LS = " << ls_number << ", Event = " << event_number;
    return retVal.str();
  }
}

void MuonPFCandidateCleaner::produce(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<MuonPFCandidateCleaner::produce>:" << std::endl;

  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  std::vector<reco::Particle::LorentzVector> selMuonP4s;
  if ( muPlus.isNonnull()  ) {
    if ( verbosity_ ) std::cout << " muPlus: Pt = " << muPlus->pt() << ", eta = " << muPlus->eta() << ", phi = " << muPlus->phi() << std::endl;
    selMuonP4s.push_back(muPlus->p4());
  }
  if ( muMinus.isNonnull() ) {
    if ( verbosity_ ) std::cout << " muMinus: Pt = " << muMinus->pt() << ", eta = " << muMinus->eta() << ", phi = " << muMinus->phi() << std::endl;
    selMuonP4s.push_back(muMinus->p4());
  }

//--- produce collection of PFCandidate excluding muons
  edm::Handle<reco::PFCandidateCollection> pfCandidates;
  evt.getByLabel(srcPFCandidates_, pfCandidates);

  std::auto_ptr<reco::PFCandidateCollection> pfCandidates_woMuons(new reco::PFCandidateCollection());   
   
//--- iterate over list of reconstructed PFCandidates, 
//    add PFCandidate to output collection in case it does not correspond to any selected muon
  std::vector<muonToPFCandMatchInfoType> selMuonToPFCandMatches;
  for ( std::vector<reco::Particle::LorentzVector>::const_iterator selMuonP4 = selMuonP4s.begin();
	selMuonP4 != selMuonP4s.end(); ++selMuonP4 ) {
    std::vector<muonToPFCandMatchInfoType> tmpMatches;
    for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates->begin();
	  pfCandidate != pfCandidates->end(); ++pfCandidate ) {
      double dR = reco::deltaR(pfCandidate->p4(), *selMuonP4);
      if ( dR < dRmatch_ ) tmpMatches.push_back(muonToPFCandMatchInfoType(*selMuonP4, &(*pfCandidate), dR));
    }
    // rank muon-to-pfCandidate matches by quality
    std::sort(tmpMatches.begin(), tmpMatches.end(), SortMuonToPFCandMatchInfosDescendingMatchQuality());
    if ( tmpMatches.size() > 0 ) selMuonToPFCandMatches.push_back(tmpMatches.front());
    if ( removeDuplicates_ ) {
      // CV: remove all high Pt charged PFCandidates very close to muon direction
      //    (duplicate tracks arise in case muon track in SiStrip + Pixel detector is reconstructed as 2 disjoint segments)
      for ( std::vector<muonToPFCandMatchInfoType>::const_iterator tmpMatch =  tmpMatches.begin();
	    tmpMatch !=  tmpMatches.end(); ++tmpMatch ) {
	if ( tmpMatch->dR_ < 1.e-3 && fabs(tmpMatch->pfCandidateCharge_) > 0.5 && tmpMatch->pfCandidatePt_ > (0.33*tmpMatch->muonPt_) ) selMuonToPFCandMatches.push_back(*tmpMatch);
      }
    }
  }

  std::vector<const reco::PFCandidate*> removedPFCandidates;
  for ( reco::PFCandidateCollection::const_iterator pfCandidate = pfCandidates->begin();
	pfCandidate != pfCandidates->end(); ++pfCandidate ) {
    bool isMuon = false;
    for ( std::vector<muonToPFCandMatchInfoType>::const_iterator muonMatchInfo = selMuonToPFCandMatches.begin();
	  muonMatchInfo != selMuonToPFCandMatches.end(); ++muonMatchInfo ) {
      if ( muonMatchInfo->pfCandidate_ == &(*pfCandidate) ) isMuon = true;
    }
    if ( verbosity_ && pfCandidate->pt() > 10. && fabs(pfCandidate->charge()) > 0.5 ) {
      std::cout << "pfCandidate: Pt = " << pfCandidate->pt() << ", eta = " << pfCandidate->eta() << ", phi = " << pfCandidate->phi() << ", isMuon = " << isMuon << std::endl;
    }
    if ( isMuon ) removedPFCandidates.push_back(&(*pfCandidate)); // pfCandidate belongs to a selected muon, do not copy
    else pfCandidates_woMuons->push_back(*pfCandidate);
  }
  if ( (removedPFCandidates.size() > selMuons.size() && numWarnings_tooMany_ < maxWarnings_tooMany_) &&
       (removedPFCandidates.size() < selMuons.size() && numWarnings_tooFew_  < maxWarnings_tooFew_ ) ) {
    edm::LogWarning("MuonPFCandidateCleaner") 
      << " (" << runLumiEventNumbers_to_string(evt) << ")" << std::endl
      << " Removed " << removedPFCandidates.size() << " PF-candidates from event containing " << selMuons.size() << " muons !!" << std::endl;
    if ( muPlus.isNonnull() ) std::cout << " muPlus: Pt = " << muPlus->pt() << ", eta = " << muPlus->eta() << ", phi = " << muPlus->phi() << std::endl;
    if ( muMinus.isNonnull() ) std::cout << " muMinus: Pt = " << muMinus->pt() << ", eta = " << muMinus->eta() << ", phi = " << muMinus->phi() << std::endl;
    int idx = 0;
    for ( std::vector<const reco::PFCandidate*>::const_iterator removedPFCandidate = removedPFCandidates.begin();
	  removedPFCandidate != removedPFCandidates.end(); ++removedPFCandidate ) {
      std::cout << "PF-candidate #" << idx << " (charge = " << (*removedPFCandidate)->charge() << "):" 
		<< " Pt = " << (*removedPFCandidate)->pt() << ", eta = " << (*removedPFCandidate)->eta() << ", phi = " << (*removedPFCandidate)->phi() << std::endl;
      ++idx;
    }
    if ( removedPFCandidates.size() > selMuons.size() ) ++numWarnings_tooMany_;
    if ( removedPFCandidates.size() < selMuons.size() ) ++numWarnings_tooFew_;
  }

  evt.put(pfCandidates_woMuons);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonPFCandidateCleaner);
