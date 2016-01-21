#include "TopQuarkAnalysis/TopEventProducers/interface/PseudoTopProducer.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "RecoJets/JetProducers/interface/JetSpecific.h"
#include "fastjet/ClusterSequence.hh"
#include "DataFormats/Math/interface/deltaR.h"

using namespace std;
using namespace edm;
using namespace reco;

PseudoTopProducer::PseudoTopProducer(const edm::ParameterSet& pset):
  finalStateToken_(consumes<edm::View<reco::Candidate> >(pset.getParameter<edm::InputTag>("finalStates"))),
  genParticleToken_(consumes<edm::View<reco::Candidate> >(pset.getParameter<edm::InputTag>("genParticles"))),
  leptonMinPt_(pset.getParameter<double>("leptonMinPt")),
  leptonMaxEta_(pset.getParameter<double>("leptonMaxEta")),
  jetMinPt_(pset.getParameter<double>("jetMinPt")),
  jetMaxEta_(pset.getParameter<double>("jetMaxEta")),
  wMass_(pset.getParameter<double>("wMass")),
  tMass_(pset.getParameter<double>("tMass"))
{
  const double leptonConeSize = pset.getParameter<double>("leptonConeSize");
  const double jetConeSize = pset.getParameter<double>("jetConeSize");
  fjLepDef_ = std::shared_ptr<JetDef>(new JetDef(fastjet::antikt_algorithm, leptonConeSize));
  fjJetDef_ = std::shared_ptr<JetDef>(new JetDef(fastjet::antikt_algorithm, jetConeSize));

  genVertex_ = reco::Particle::Point(0,0,0);

  produces<reco::GenParticleCollection>("neutrinos");
  produces<reco::GenJetCollection>("leptons");
  produces<reco::GenJetCollection>("jets");
  produces<reco::METCollection>("mets");

  produces<reco::GenParticleCollection>();

}

void PseudoTopProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{
  edm::Handle<edm::View<reco::Candidate> > finalStateHandle;
  event.getByToken(finalStateToken_, finalStateHandle);

  edm::Handle<edm::View<reco::Candidate> > genParticleHandle;
  event.getByToken(genParticleToken_, genParticleHandle);

  std::auto_ptr<reco::GenParticleCollection> neutrinos(new reco::GenParticleCollection);
  std::auto_ptr<reco::GenJetCollection> leptons(new reco::GenJetCollection);
  std::auto_ptr<reco::GenJetCollection> jets(new reco::GenJetCollection);
  std::auto_ptr<reco::METCollection> mets(new reco::METCollection);
  auto neutrinosRefHandle = event.getRefBeforePut<reco::GenParticleCollection>("neutrinos");
  auto leptonsRefHandle = event.getRefBeforePut<reco::GenJetCollection>("leptons");
  auto jetsRefHandle = event.getRefBeforePut<reco::GenJetCollection>("jets");

  std::auto_ptr<reco::GenParticleCollection> pseudoTop(new reco::GenParticleCollection);
  auto pseudoTopRefHandle = event.getRefBeforePut<reco::GenParticleCollection>();

  // Collect unstable B-hadrons
  std::set<size_t> bHadronIdxs;
  for ( size_t i=0, n=genParticleHandle->size(); i<n; ++i )
  {
    const reco::Candidate& p = genParticleHandle->at(i);
    const int status = p.status();
    if ( status == 1 ) continue;

    // Collect B-hadrons, to be used in b tagging
    if ( isBHadron(&p) ) bHadronIdxs.insert(i);
  }

  // Collect stable leptons and neutrinos
  size_t nStables = 0;
  std::vector<size_t> leptonIdxs;
  double metX = 0, metY = 0;
  for ( size_t i=0, n=finalStateHandle->size(); i<n; ++i )
  {
    const reco::Candidate& p = finalStateHandle->at(i);
    const int absPdgId = abs(p.pdgId());
    if ( p.status() != 1 ) continue;

    ++nStables;
    if ( p.numberOfMothers() == 0 ) continue; // Skip orphans (if exists)
    if ( p.mother()->status() == 4 ) continue; // Treat particle as hadronic if directly from the incident beam (protect orphans in MINIAOD)
    if ( isHadron(absPdgId) or isFromHadron(&p) ) continue;
    switch ( absPdgId )
    {
      case 11: case 13: // Leptons
      case 22: // Photons
        leptonIdxs.push_back(i);
        break;
      case 12: case 14: case 16:
        metX += p.px();
        metY += p.py();
        neutrinos->push_back(reco::GenParticle(p.charge(), p.p4(), p.vertex(), p.pdgId(), p.status(), true));
        break;
    }
  }

  // Sort neutrinos by pT.
  std::sort(neutrinos->begin(), neutrinos->end(), GreaterByPt<reco::Candidate>());
  mets->push_back(reco::MET(LorentzVector(metX, metY, 0, std::hypot(metX, metY)), genVertex_));

  // Make dressed leptons with anti-kt(0.1) algorithm
  //// Prepare input particle list
  std::vector<fastjet::PseudoJet> fjLepInputs;
  fjLepInputs.reserve(leptonIdxs.size());
  for ( auto index : leptonIdxs )
  {
    const reco::Candidate& p = finalStateHandle->at(index);
    if ( std::isnan(p.pt()) or p.pt() <= 0 ) continue;

    fjLepInputs.push_back(fastjet::PseudoJet(p.px(), p.py(), p.pz(), p.energy()));
    fjLepInputs.back().set_user_index(index);
  }

  //// Run the jet algorithm
  fastjet::ClusterSequence fjLepClusterSeq(fjLepInputs, *fjLepDef_);
  std::vector<fastjet::PseudoJet> fjLepJets = fastjet::sorted_by_pt(fjLepClusterSeq.inclusive_jets(leptonMinPt_));

  //// Build dressed lepton objects from the FJ output
  leptons->reserve(fjLepJets.size());
  std::set<size_t> lepDauIdxs; // keep lepton constituents to remove from GenJet construction
  for ( auto& fjJet : fjLepJets )
  {
    if ( abs(fjJet.eta()) > leptonMaxEta_ ) continue;

    // Get jet constituents from fastJet
    const std::vector<fastjet::PseudoJet> fjConstituents = fastjet::sorted_by_pt(fjJet.constituents());
    // Convert to CandidatePtr
    std::vector<reco::CandidatePtr> constituents;
    reco::CandidatePtr lepCand;
    for ( auto& fjConstituent : fjConstituents )
    {
      const size_t index = fjConstituent.user_index();
      reco::CandidatePtr cand = finalStateHandle->ptrAt(index);
      const int absPdgId = abs(cand->pdgId());
      if ( absPdgId == 11 or absPdgId == 13 )
      {
        if ( lepCand.isNonnull() and lepCand->pt() > cand->pt() ) continue; // Choose one with highest pt
        lepCand = cand;
      }
      constituents.push_back(cand);
    }
    if ( lepCand.isNull() ) continue;
    //if ( lepCand->pt() < fjJet.pt()/2 ) continue; // Central lepton must be the major component

    const LorentzVector jetP4(fjJet.px(), fjJet.py(), fjJet.pz(), fjJet.E());
    reco::GenJet lepJet;
    reco::writeSpecific(lepJet, jetP4, genVertex_, constituents, eventSetup);

    lepJet.setPdgId(lepCand->pdgId());
    lepJet.setCharge(lepCand->charge());

    const double jetArea = fjJet.has_area() ? fjJet.area() : 0;
    lepJet.setJetArea(jetArea);

    leptons->push_back(lepJet);

    // Keep constituent indices to be used in the next step.
    for ( auto& fjConstituent : fjConstituents )
    {
      lepDauIdxs.insert(fjConstituent.user_index());
    }
  }

  // Now proceed to jets.
  // Jets: anti-kt excluding the e, mu, nu, and photons in selected leptons.
  //// Prepare input particle list. Remove particles used in lepton clusters, neutrinos
  std::vector<fastjet::PseudoJet> fjJetInputs;
  fjJetInputs.reserve(nStables);
  for ( size_t i=0, n=finalStateHandle->size(); i<n; ++i )
  {
    const reco::Candidate& p = finalStateHandle->at(i);
    if ( p.status() != 1 ) continue;
    if ( std::isnan(p.pt()) or p.pt() <= 0 ) continue;
    const int aid = abs(p.pdgId());
    if ( aid == 12 or aid == 14 or aid == 16 ) continue;

    if ( lepDauIdxs.find(i) != lepDauIdxs.end() ) continue;

    fjJetInputs.push_back(fastjet::PseudoJet(p.px(), p.py(), p.pz(), p.energy()));
    fjJetInputs.back().set_user_index(i);
  }

  //// Also don't forget to put B hadrons
  for ( auto index : bHadronIdxs )
  {
    const reco::Candidate& p = genParticleHandle->at(index);
    if ( std::isnan(p.pt()) or p.pt() <= 0 ) continue;

    const double scale = 1e-20/p.p();
    fjJetInputs.push_back(fastjet::PseudoJet(p.px()*scale, p.py()*scale, p.pz()*scale, p.energy()*scale));
    fjJetInputs.back().set_user_index(index);
  }

  //// Run the jet algorithm
  fastjet::ClusterSequence fjJetClusterSeq(fjJetInputs, *fjJetDef_);
  std::vector<fastjet::PseudoJet> fjJets = fastjet::sorted_by_pt(fjJetClusterSeq.inclusive_jets(jetMinPt_));

  /// Build jets
  jets->reserve(fjJets.size());
  std::vector<size_t> bjetIdxs, ljetIdxs;
  for ( auto& fjJet : fjJets )
  {
    if ( abs(fjJet.eta()) > jetMaxEta_ ) continue;

    // Get jet constituents from fastJet
    const std::vector<fastjet::PseudoJet> fjConstituents = fastjet::sorted_by_pt(fjJet.constituents());
    // Convert to CandidatePtr
    std::vector<reco::CandidatePtr> constituents;
    bool hasBHadron = false;
    for ( size_t j=0, m=fjConstituents.size(); j<m; ++j )
    {
      const size_t index = fjConstituents[j].user_index();
      if ( bHadronIdxs.find(index) != bHadronIdxs.end() ) hasBHadron = true;
      reco::CandidatePtr cand;
      if ( bHadronIdxs.find(index) != bHadronIdxs.end() )
      {
        hasBHadron = true;
        cand = genParticleHandle->ptrAt(index);
      }
      else cand = finalStateHandle->ptrAt(index);
      if ( cand.isNull() ) continue;
      constituents.push_back(cand);
    }

    const LorentzVector jetP4(fjJet.px(), fjJet.py(), fjJet.pz(), fjJet.E());
    reco::GenJet genJet;
    reco::writeSpecific(genJet, jetP4, genVertex_, constituents, eventSetup);

    const double jetArea = fjJet.has_area() ? fjJet.area() : 0;
    genJet.setJetArea(jetArea);
    if ( hasBHadron )
    {
      genJet.setPdgId(5);
      bjetIdxs.push_back(jets->size());
    }
    else
    {
      ljetIdxs.push_back(jets->size());
    }

    jets->push_back(genJet);
  }

  // Every building blocks are ready. Continue to pseudo-W and pseudo-top combination
  // NOTE : A C++ trick, use do-while instead of long-nested if-statements.
  do
  {
    if ( bjetIdxs.size() < 2 ) break; // Ignore single top for now.

    int w1Q = 1, w2Q = -1;
    int w1dau1Id = 1, w2dau1Id = -1;
    LorentzVector w1dau1LVec, w1dau2LVec;
    LorentzVector w2dau1LVec, w2dau2LVec;

    // Determine channel by the number of leptons
    const size_t nLep = leptons->size();
    if ( nLep == 2 ) { // Dilepton channel first
      const size_t nNu = neutrinos->size();
      if ( nNu < 2 ) break; // Skip this bizarre event

      const auto& lepton1 = leptons->at(0);
      const auto& lepton2 = leptons->at(1);
      w1dau1LVec = lepton1.p4();
      w2dau1LVec = lepton2.p4();
      const double zM = (w1dau1LVec+w2dau1LVec).mass();
      if ( zM < 20 ) break;
      w1Q = lepton1.charge();
      w2Q = lepton2.charge();
      if ( w1Q+w2Q != 0 ) break;
      w1dau1Id = lepton1.pdgId();
      w2dau1Id = lepton2.pdgId();

      double minDm = 1e9;
      int selNu1 = -1, selNu2 = -1;
      for ( size_t iNu=0; iNu<nNu; ++iNu ) {
        const double dm1 = std::abs((w1dau1LVec+neutrinos->at(iNu).p4()).mass()-wMass_);
        for ( size_t jNu=0; jNu<nNu; ++jNu ) {
          if ( iNu == jNu ) continue;

          const double dm2 = std::abs((w2dau1LVec+neutrinos->at(jNu).p4()).mass()-wMass_);
          const double dmSum = dm1 + dm2;

          if ( dmSum < minDm ) {
            minDm = dmSum;
            selNu1 = iNu;
            selNu2 = jNu;
          }
        }
      }
      if ( selNu1 < 0 or selNu2 < 0 ) break; // Should never happen, but for safety...

      // Now we have best pair for the dilepton channel
      w1dau2LVec = neutrinos->at(selNu1).p4();
      w2dau2LVec = neutrinos->at(selNu2).p4();
    }
    else if ( nLep == 1 ) { // Continue to the semi leptonic channel
      const size_t nNu = neutrinos->size();
      const size_t nLjet = ljetIdxs.size();
      if ( nNu < 1 or nLjet < 2 ) break; // Skip this bizarre event. We need at least one neutrino and two light flavour jets to build two W candidates

      const auto& lepton = leptons->at(0);
      const auto lepLVec = lepton.p4();

      double minDm = 1e9;
      int selNu = -1;
      for ( size_t iNu=0; iNu<nNu; ++iNu ) {
        const double dm = std::abs((lepLVec+neutrinos->at(iNu).p4()).mass()-wMass_);
        if ( dm < minDm ) {
          minDm = dm;
          selNu = iNu;
        }
      }
      if ( selNu < 0 ) break; // Never happens, but for safety

      minDm = 1e9;
      int selJet1 = -1, selJet2 = -1;
      for ( size_t ii=0; ii<nLjet; ++ii ) {
        const size_t i = ljetIdxs[ii];
        const auto ljet1LVec = jets->at(i).p4();
        if ( deltaR(ljet1LVec, lepLVec) < 0.4 ) continue;
        for ( size_t jj=ii+1; jj<nLjet; ++jj )
        {
          const size_t j = ljetIdxs[jj];
          const auto ljet2LVec = jets->at(j).p4();
          if ( deltaR(ljet2LVec, lepLVec) < 0.4 ) continue;

          const double dm = std::abs((ljet1LVec+ljet2LVec).mass()-wMass_);
          if ( dm < minDm ) {
            minDm = dm;
            selJet1 = i;
            selJet2 = j;
          }
        }
      }
      if ( selJet1 < 0 or selJet2 < 0 ) break; // Skip this event if we cannot find hadronic W candidate (can happend due to dR removal)

      // Now we have best pair for the semileptonic channel
      w1dau1LVec = lepLVec;
      w1dau2LVec = neutrinos->at(selNu).p4();
      w2dau1LVec = jets->at(selJet1).p4();
      w2dau2LVec = jets->at(selJet2).p4();
      w1Q = lepton.charge();
      w2Q = -w1Q;
      w1dau1Id = lepton.pdgId();
    }

    const auto w1LVec = w1dau1LVec+w1dau2LVec;
    const auto w2LVec = w2dau1LVec+w2dau2LVec;

    // Combine b jets
    size_t bjetIdx1 = 999, bjetIdx2 = 999;
    double sumDm = 1e9;
    for ( size_t i : bjetIdxs )
    {
      const auto& bjet1 = jets->at(i);
      if ( nLep > 0 and deltaR(bjet1.p4(), leptons->at(0).p4()) < 0.4 ) continue;
      if ( nLep > 1 and deltaR(bjet1.p4(), leptons->at(1).p4()) < 0.4 ) continue;

      const double mtop1 = (w1LVec+bjet1.p4()).mass();
      const double dmtop1 = std::abs(mtop1-tMass_);
      for ( size_t j : bjetIdxs )
      {
        if ( i == j ) continue;
        const auto& bjet2 = jets->at(j);
        if ( nLep > 0 and deltaR(bjet2.p4(), leptons->at(0).p4()) < 0.4 ) continue;
        if ( nLep > 1 and deltaR(bjet2.p4(), leptons->at(1).p4()) < 0.4 ) continue;

        const double mtop2 = (w2LVec+bjet2.p4()).mass();
        const double dmtop2 = std::abs(mtop2-tMass_);

        if ( sumDm <= dmtop1+dmtop2 ) continue;

        sumDm = dmtop1+dmtop2;
        bjetIdx1 = i;
        bjetIdx2 = j;
      }
    }
    if ( sumDm >= 1e9 ) break; // Failed to make top, but this should not happen.

    const auto& b1LVec = jets->at(bjetIdx1).p4();
    const auto& b2LVec = jets->at(bjetIdx2).p4();
    const auto t1LVec = w1LVec + b1LVec;
    const auto t2LVec = w2LVec + b2LVec;

    // Put all of them into candidate collection
    // t->wb, w->pq
    reco::GenParticle t1(w1Q*2/3, t1LVec, genVertex_, w1Q*6, 3, false);
    reco::GenParticle w1(w1Q, w1LVec, genVertex_, w1Q*24, 3, true);
    reco::GenParticle b1(0, b1LVec, genVertex_, w1Q*5, 1, true);
    reco::GenParticle p1(w1Q, w1dau1LVec, genVertex_, w1dau1Id, 1, true);
    reco::GenParticle q1(0, w1dau2LVec, genVertex_, -w1dau1Id+w1Q, 1, true);

    reco::GenParticle t2(w2Q*2/3, t2LVec, genVertex_, w2Q*6, 3, false);
    reco::GenParticle w2(w2Q, w2LVec, genVertex_, w2Q*24, 3, true);
    reco::GenParticle b2(0, b2LVec, genVertex_, w2Q*5, 1, true);
    reco::GenParticle p2(w2Q, w2dau1LVec, genVertex_, w2dau1Id, 1, true);
    reco::GenParticle q2(0, w2dau2LVec, genVertex_, -w2dau1Id+w2Q, 1, true);

    pseudoTop->push_back(t1);
    pseudoTop->push_back(t2);

    pseudoTop->push_back(w1);
    pseudoTop->push_back(b1);

    pseudoTop->push_back(w2);
    pseudoTop->push_back(b2);

    pseudoTop->push_back(p1);
    pseudoTop->push_back(q1);

    pseudoTop->push_back(p2);
    pseudoTop->push_back(q2);

    // t->W+b, tbar->W-b
    pseudoTop->at(0).addDaughter(reco::GenParticleRef(pseudoTopRefHandle, 2)); // t->W
    pseudoTop->at(0).addDaughter(reco::GenParticleRef(pseudoTopRefHandle, 3)); // t->b
    pseudoTop->at(2).addMother(reco::GenParticleRef(pseudoTopRefHandle, 0)); // t->W
    pseudoTop->at(3).addMother(reco::GenParticleRef(pseudoTopRefHandle, 0)); // t->b

    pseudoTop->at(1).addDaughter(reco::GenParticleRef(pseudoTopRefHandle, 4));
    pseudoTop->at(1).addDaughter(reco::GenParticleRef(pseudoTopRefHandle, 5));
    pseudoTop->at(4).addMother(reco::GenParticleRef(pseudoTopRefHandle, 1));
    pseudoTop->at(5).addMother(reco::GenParticleRef(pseudoTopRefHandle, 1));

    // W->lv or W->jj
    pseudoTop->at(2).addDaughter(reco::GenParticleRef(pseudoTopRefHandle, 6));
    pseudoTop->at(2).addDaughter(reco::GenParticleRef(pseudoTopRefHandle, 7));
    pseudoTop->at(6).addMother(reco::GenParticleRef(pseudoTopRefHandle, 2));
    pseudoTop->at(7).addMother(reco::GenParticleRef(pseudoTopRefHandle, 2));

    pseudoTop->at(4).addDaughter(reco::GenParticleRef(pseudoTopRefHandle, 8));
    pseudoTop->at(4).addDaughter(reco::GenParticleRef(pseudoTopRefHandle, 9));
    pseudoTop->at(8).addMother(reco::GenParticleRef(pseudoTopRefHandle, 4));
    pseudoTop->at(9).addMother(reco::GenParticleRef(pseudoTopRefHandle, 4));
  } while (false);

  event.put(neutrinos, "neutrinos");
  event.put(leptons, "leptons");
  event.put(jets, "jets");
  event.put(mets, "mets");

  event.put(pseudoTop);
}

const reco::Candidate* PseudoTopProducer::getLast(const reco::Candidate* p)
{
  for ( size_t i=0, n=p->numberOfDaughters(); i<n; ++i )
  {
    const reco::Candidate* dau = p->daughter(i);
    if ( p->pdgId() == dau->pdgId() ) return getLast(dau);
  }
  return p;
}

void PseudoTopProducer::insertAllDaughters(const reco::Candidate* p, std::set<const reco::Candidate*>& list) const
{
  if ( !p ) return;
  if ( list.find(p) != list.end() ) return; // Skip if already in the list

  list.insert(p);
  for ( size_t i=0, n=p->numberOfDaughters(); i<n; ++i )
  {
    const reco::Candidate* dau = p->daughter(i);
    list.insert(dau);
    insertAllDaughters(dau, list);
  }
}

bool PseudoTopProducer::isHadron(const int pdgId) const
{
  const unsigned int aid = abs(pdgId);
  if ( aid/10000000 > 0 ) return false; // particle with extra bit is not a hadron
  if ( aid <= 100 ) return false; // Hadrons have pdgId > 100

  const unsigned char nj  = aid % 10;
  const unsigned char nq3 = (aid/10) % 10;
  const unsigned char nq2 = (aid/100) % 10;
  const unsigned char nq1 = (aid/1000) % 10;

  // Check for penta quarks, 9 at the 8th digit
  if ( (aid/1000000) % 10 == 9 )
  {
    const unsigned char nl  = (aid/10000) % 10;
    const unsigned char nr  = (aid/100000) % 10;

    if ( nr != 9 && nr != 0 && nj != 9 && nj != 0 &&
         nq1+nq2+nq3 != 0 &&
         nq2 <= nq1 && nq1 <= nl && nl <= nr ) return true;
  }

  // Continue to usual mesons and baryons
  const int fundID = (nq1+nq2 == 0) ? aid%10000 : 0;
  if( fundID <= 100 && fundID > 0 ) return false;

  const std::set<unsigned int> hadronIds = {
    130, 310, 210, // ordinary mesons
    150, 350, 510, 530, // EvtGen specific numbers
    110, 990, 9990, // pomerons, etc
    2110, 2210, // Baryons
  };
  if ( hadronIds.find(aid) != hadronIds.end() ) return true;

  if( nj > 0 && nq3 > 0 && nq2 > 0 )
  {
    if ( nq1 == 0 && (nq3 != nq2 || pdgId >= 0) ) return true;
    if ( nq1 > 0 ) return true;
  }

  return false;
}

bool PseudoTopProducer::isFromHadron(const reco::Candidate* p) const
{
  for ( int i=0, n=p->numberOfMothers(); i<n; ++i )
  {
    const reco::Candidate* mother = p->mother(i);
    if ( !mother or mother->numberOfMothers() == 0 ) continue; // reaches to incident beam or top level
    if ( isHadron(mother->pdgId()) or isFromHadron(mother) ) return true;
  }

  return false;
}

bool PseudoTopProducer::isBHadron(const reco::Candidate* p) const
{
  const unsigned int absPdgId = abs(p->pdgId());
  if ( !isBHadron(absPdgId) ) return false;

  // Do not consider this particle if it has B hadron daughter
  // For example, B* -> B0 + photon; then we drop B* and take B0 only
  for ( int i=0, n=p->numberOfDaughters(); i<n; ++i )
  {
    const reco::Candidate* dau = p->daughter(i);
    if ( isBHadron(abs(dau->pdgId())) ) return false;
  }

  return true;
}

bool PseudoTopProducer::isBHadron(const unsigned int absPdgId) const
{
  if ( absPdgId <= 100 ) return false; // Fundamental particles and MC internals
  if ( absPdgId >= 1000000000 ) return false; // Nuclei, +-10LZZZAAAI

  // General form of PDG ID is 7 digit form
  // +- n nr nL nq1 nq2 nq3 nJ
  //const int nJ = absPdgId % 10; // Spin
  const int nq3 = (absPdgId / 10) % 10;
  const int nq2 = (absPdgId / 100) % 10;
  const int nq1 = (absPdgId / 1000) % 10;

  if ( nq3 == 0 ) return false; // Diquarks
  if ( nq1 == 0 and nq2 == 5 ) return true; // B mesons
  if ( nq1 == 5 ) return true; // B baryons

  return false;
}

reco::GenParticleRef PseudoTopProducer::buildGenParticle(const reco::Candidate* p, reco::GenParticleRefProd& refHandle,
                                                         std::auto_ptr<reco::GenParticleCollection>& outColl) const
{
  reco::GenParticle pOut(*dynamic_cast<const reco::GenParticle*>(p));
  pOut.clearMothers();
  pOut.clearDaughters();
  pOut.resetMothers(refHandle.id());
  pOut.resetDaughters(refHandle.id());

  outColl->push_back(pOut);

  return reco::GenParticleRef(refHandle, outColl->size()-1);
}

