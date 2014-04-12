#ifndef TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h
#define TauAnalysis_MCEmbeddingTools_embeddingAuxFunctions_h

/**
 *
 * Define methods to retrieve "the" (= highest Pt)
 * muons of positive and negative charge from the event.
 * 
 * \author Christian Veelken, LLR
 *
 * \version $Revision: 1.9 $
 *
 * $Id: embeddingAuxFunctions.h,v 1.9 2013/02/05 20:01:19 veelken Exp $
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetMatchInfo.h"

#include "HepMC/IO_HEPEVT.h"

std::vector<reco::CandidateBaseRef> getSelMuons(const edm::Event&, const edm::InputTag&);

reco::CandidateBaseRef getTheMuPlus(const std::vector<reco::CandidateBaseRef>&);
reco::CandidateBaseRef getTheMuMinus(const std::vector<reco::CandidateBaseRef>&);

TrackDetMatchInfo getTrackDetMatchInfo(const edm::Event&, const edm::EventSetup&, TrackDetectorAssociator&, const TrackAssociatorParameters&, const reco::Candidate*);

bool matchMuonDetId(uint32_t, uint32_t);
void printMuonDetId(const edm::EventSetup&, uint32_t);

void repairBarcodes(HepMC::GenEvent*);

const reco::GenParticle* findGenParticleForMCEmbedding(const reco::Candidate::LorentzVector&, const reco::GenParticleCollection&, double, int, const std::vector<int>*, bool);

void compGenMuonP4afterRad(const reco::GenParticle*, reco::Candidate::LorentzVector&);
void compGenTauP4afterRad(const reco::GenParticle*, reco::Candidate::LorentzVector&);

void findMuons(const edm::Event&, const edm::InputTag&, reco::Candidate::LorentzVector&, bool&, reco::Candidate::LorentzVector&, bool&);

double getDeDxForPbWO4(double);

const double DENSITY_PBWO4 = 8.28;
const double DENSITY_BRASS = 8.53;
const double DENSITY_IRON  = 7.87;

#endif
