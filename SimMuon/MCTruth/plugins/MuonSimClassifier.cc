// -*- C++ -*-
//
// Package:    SimMuon/MCTruth
// Class:      MuonSimClassifier
//
/*


 CLASSIFICATION: For each RECO Muon, match to SIM particle, and then:
  - If the SIM is not a Muon, label as Punchthrough (1) except if it is an
 electron or positron (11)
  - If the SIM is a Muon, then look at it's provenance.
     A) the SIM muon is also a GEN muon, whose parent is NOT A HADRON AND NOT A
 TAU
        -> classify as "primary" (4).
     B) the SIM muon is also a GEN muon, whose parent is HEAVY FLAVOURED HADRON
 OR A TAU
        -> classify as "heavy flavour" (3)
     C) classify as "light flavour/decay" (2)

  In any case, if the TP is not preferentially matched back to the same RECO
 muon, label as Ghost (flip the classification)


 FLAVOUR:
  - for non-muons: 0
  - for primary muons: 13
  - for non primary muons: flavour of the mother: std::abs(pdgId) of heaviest
 quark, or 15 for tau

*/
//
// Original Author:  G.Petrucciani and G.Abbiendi
//         Created:  Sun Nov 16 16:14:09 CET 2008
//         revised:  3/Aug/2017
//

// system include files
#include <memory>
#include <set>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSimInfo.h"

#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertex.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

//
// class decleration
class MuonSimClassifier : public edm::stream::EDProducer<> {
public:
  explicit MuonSimClassifier(const edm::ParameterSet &);
  ~MuonSimClassifier() override;

private:
  void produce(edm::Event &, const edm::EventSetup &) override;
  /// The RECO objects
  edm::EDGetTokenT<edm::View<reco::Muon>> muonsToken_;

  /// Track to use
  reco::MuonTrackType trackType_;

  /// The TrackingParticle objects
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticlesToken_;

  /// The Associations
  edm::InputTag associatorLabel_;
  edm::EDGetTokenT<reco::MuonToTrackingParticleAssociator> muAssocToken_;

  /// Cylinder to use to decide if a decay is early or late
  double decayRho_, decayAbsZ_;

  /// Create a link to the generator level particles
  bool linkToGenParticles_;
  edm::InputTag genParticles_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticlesToken_;

  /// Returns the flavour given a pdg id code
  int flavour(int pdgId) const;

  /// Write a ValueMap<int> in the event
  template <typename T>
  void writeValueMap(edm::Event &iEvent,
                     const edm::Handle<edm::View<reco::Muon>> &handle,
                     const std::vector<T> &values,
                     const std::string &label) const;

  TrackingParticleRef getTpMother(TrackingParticleRef tp) {
    if (tp.isNonnull() && tp->parentVertex().isNonnull() && !tp->parentVertex()->sourceTracks().empty()) {
      return tp->parentVertex()->sourceTracks()[0];
    } else {
      return TrackingParticleRef();
    }
  }

  /// Convert TrackingParticle into GenParticle, save into output collection,
  /// if mother is primary set reference to it,
  /// return index in output collection
  int convertAndPush(const TrackingParticle &tp,
                     reco::GenParticleCollection &out,
                     const TrackingParticleRef &momRef,
                     const edm::Handle<reco::GenParticleCollection> &genParticles) const;
};

MuonSimClassifier::MuonSimClassifier(const edm::ParameterSet &iConfig)
    : muonsToken_(consumes<edm::View<reco::Muon>>(iConfig.getParameter<edm::InputTag>("muons"))),
      trackingParticlesToken_(
          consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("trackingParticles"))),
      muAssocToken_(
          consumes<reco::MuonToTrackingParticleAssociator>(iConfig.getParameter<edm::InputTag>("associatorLabel"))),
      decayRho_(iConfig.getParameter<double>("decayRho")),
      decayAbsZ_(iConfig.getParameter<double>("decayAbsZ")),
      linkToGenParticles_(iConfig.getParameter<bool>("linkToGenParticles")),
      genParticles_(linkToGenParticles_ ? iConfig.getParameter<edm::InputTag>("genParticles") : edm::InputTag())

{
  std::string trackType = iConfig.getParameter<std::string>("trackType");
  if (trackType == "inner")
    trackType_ = reco::InnerTk;
  else if (trackType == "outer")
    trackType_ = reco::OuterTk;
  else if (trackType == "global")
    trackType_ = reco::GlobalTk;
  else if (trackType == "segments")
    trackType_ = reco::Segments;
  else if (trackType == "glb_or_trk")
    trackType_ = reco::GlbOrTrk;
  else
    throw cms::Exception("Configuration") << "Track type '" << trackType << "' not supported.\n";
  if (linkToGenParticles_) {
    genParticlesToken_ = consumes<reco::GenParticleCollection>(genParticles_);
  }

  produces<edm::ValueMap<reco::MuonSimInfo>>();
  if (linkToGenParticles_) {
    produces<reco::GenParticleCollection>("secondaries");
    produces<edm::Association<reco::GenParticleCollection>>("toPrimaries");
    produces<edm::Association<reco::GenParticleCollection>>("toSecondaries");
  }
}

MuonSimClassifier::~MuonSimClassifier() {}

void dumpFormatedInfo(const reco::MuonSimInfo &simInfo) {
  return;
  LogTrace("MuonSimClassifier") << "\t Particle pdgId = " << simInfo.pdgId << ", (Event,Bx) = "
                                << "(" << simInfo.tpEvent << "," << simInfo.tpBX << ")"
                                << "\n\t   q*p = " << simInfo.charge * simInfo.p4.P() << ", pT = " << simInfo.p4.pt()
                                << ", eta = " << simInfo.p4.eta() << ", phi = " << simInfo.p4.phi()
                                << "\n\t   produced at vertex rho = " << simInfo.vertex.Rho()
                                << ", z = " << simInfo.vertex.Z() << ", (GEANT4 process = " << simInfo.g4processType
                                << ")\n";
}

void MuonSimClassifier::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<edm::View<reco::Muon>> muons;
  iEvent.getByToken(muonsToken_, muons);

  edm::Handle<TrackingParticleCollection> trackingParticles;
  iEvent.getByToken(trackingParticlesToken_, trackingParticles);

  edm::Handle<reco::GenParticleCollection> genParticles;
  if (linkToGenParticles_) {
    iEvent.getByToken(genParticlesToken_, genParticles);
  }

  edm::Handle<reco::MuonToTrackingParticleAssociator> associatorBase;
  iEvent.getByToken(muAssocToken_, associatorBase);
  const reco::MuonToTrackingParticleAssociator *assoByHits = associatorBase.product();

  reco::MuonToSimCollection recSimColl;
  reco::SimToMuonCollection simRecColl;
  LogTrace("MuonSimClassifier") << "\n "
                                   "***************************************************************** ";
  LogTrace("MuonSimClassifier") << " RECO MUON association, type:  " << trackType_;
  LogTrace("MuonSimClassifier") << " ******************************************"
                                   "*********************** \n";

  edm::RefToBaseVector<reco::Muon> allMuons;
  for (size_t i = 0, n = muons->size(); i < n; ++i) {
    allMuons.push_back(muons->refAt(i));
  }

  edm::RefVector<TrackingParticleCollection> allTPs;
  for (size_t i = 0, n = trackingParticles->size(); i < n; ++i) {
    allTPs.push_back(TrackingParticleRef(trackingParticles, i));
  }

  assoByHits->associateMuons(recSimColl, simRecColl, allMuons, trackType_, allTPs);

  // for global muons without hits on muon detectors, look at the linked
  // standalone muon
  reco::MuonToSimCollection updSTA_recSimColl;
  reco::SimToMuonCollection updSTA_simRecColl;
  if (trackType_ == reco::GlobalTk) {
    LogTrace("MuonSimClassifier") << "\n "
                                     "***************************************************************** ";
    LogTrace("MuonSimClassifier") << " STANDALONE (UpdAtVtx) MUON association ";
    LogTrace("MuonSimClassifier") << " ****************************************"
                                     "************************* \n";
    assoByHits->associateMuons(updSTA_recSimColl, updSTA_simRecColl, allMuons, reco::OuterTk, allTPs);
  }

  typedef reco::MuonToSimCollection::const_iterator r2s_it;
  typedef reco::SimToMuonCollection::const_iterator s2r_it;

  size_t nmu = muons->size();
  LogTrace("MuonSimClassifier") << "\n There are " << nmu << " reco::Muons.";

  std::vector<reco::MuonSimInfo> simInfo;

  std::unique_ptr<reco::GenParticleCollection> secondaries;  // output collection of secondary muons
  std::map<TrackingParticleRef, int> tpToSecondaries;        // map from tp to (index+1) in output collection
  std::vector<int> muToPrimary(nmu, -1), muToSecondary(nmu,
                                                       -1);  // map from input into (index) in output, -1 for null
  if (linkToGenParticles_)
    secondaries.reset(new reco::GenParticleCollection());

  // loop on reco muons
  for (size_t i = 0; i < nmu; ++i) {
    simInfo.push_back(reco::MuonSimInfo());
    LogTrace("MuonSimClassifier") << "\n reco::Muon # " << i;

    TrackingParticleRef tp;
    edm::RefToBase<reco::Muon> muMatchBack;
    r2s_it match = recSimColl.find(allMuons.at(i));
    s2r_it matchback;
    if (match != recSimColl.end()) {
      // match->second is vector, front is first element, first is the ref
      // (second would be the quality)
      tp = match->second.front().first;
      simInfo[i].tpId = tp.isNonnull() ? tp.key() : -1;  // we check, even if null refs should not appear here at all
      simInfo[i].tpAssoQuality = match->second.front().second;
      s2r_it matchback = simRecColl.find(tp);
      if (matchback != simRecColl.end()) {
        muMatchBack = matchback->second.front().first;
      } else {
        LogTrace("MuonSimClassifier") << "\n***WARNING:  This I do NOT understand: why no match back? "
                                         "*** \n";
      }
    } else {
      if ((trackType_ == reco::GlobalTk) && allMuons.at(i)->isGlobalMuon()) {
        // perform a second attempt, matching with the standalone muon
        r2s_it matchSta = updSTA_recSimColl.find(allMuons.at(i));
        if (matchSta != updSTA_recSimColl.end()) {
          tp = matchSta->second.front().first;
          simInfo[i].tpId = tp.isNonnull() ? tp.key() : -1;  // we check, even if null refs
                                                             // should not appear here at all
          simInfo[i].tpAssoQuality = matchSta->second.front().second;
          s2r_it matchback = updSTA_simRecColl.find(tp);
          if (matchback != updSTA_simRecColl.end()) {
            muMatchBack = matchback->second.front().first;
          } else {
            LogTrace("MuonSimClassifier") << "\n***WARNING:  This I do NOT understand: why no match back "
                                             "in updSTA? *** \n";
          }
        }
      } else {
        LogTrace("MuonSimClassifier") << "\t No matching TrackingParticle is found ";
      }
    }

    if (tp.isNonnull()) {
      bool isGhost = muMatchBack != allMuons.at(i);
      if (isGhost)
        LogTrace("MuonSimClassifier") << "\t *** This seems a Duplicate muon ! "
                                         "classif[i] will be < 0 ***";

      // identify signal and pileup TP
      simInfo[i].tpBX = tp->eventId().bunchCrossing();
      simInfo[i].tpEvent = tp->eventId().event();

      simInfo[i].pdgId = tp->pdgId();
      simInfo[i].vertex = tp->vertex();

      // added info on GEANT process producing the TrackingParticle
      const std::vector<SimVertex> &g4Vs = tp->parentVertex()->g4Vertices();
      simInfo[i].g4processType = g4Vs[0].processType();

      simInfo[i].charge = tp->charge();
      simInfo[i].p4 = tp->p4();

      // Try to extract mother and grand mother of this muon.
      // Unfortunately, SIM and GEN histories require diffent code :-(
      if (!tp->genParticles().empty()) {  // Muon is in GEN
        reco::GenParticleRef genp = tp->genParticles()[0];
        reco::GenParticleRef genMom = genp->numberOfMothers() > 0 ? genp->motherRef() : reco::GenParticleRef();
        reco::GenParticleRef mMom = genMom;

        if (genMom.isNonnull()) {
          if (genMom->pdgId() != tp->pdgId()) {
            simInfo[i].motherPdgId = genMom->pdgId();
            simInfo[i].motherStatus = genMom->status();
            simInfo[i].motherVertex = genMom->vertex();
          } else {
            // if mother has the same identity look backwards for the real
            // mother (it may happen in radiative decays)
            int jm = 0;
            while (mMom->pdgId() == tp->pdgId()) {
              jm++;
              if (mMom->numberOfMothers() > 0) {
                mMom = mMom->motherRef();
              } else {
                LogTrace("MuonSimClassifier") << "\t No Mother is found ";
                break;
              }

              LogTrace("MuonSimClassifier") << "\t\t backtracking mother " << jm << ", pdgId = " << mMom->pdgId()
                                            << ", status= " << mMom->status();
            }
            genMom = mMom;  // redefine genMom
            simInfo[i].motherPdgId = genMom->pdgId();
            simInfo[i].motherStatus = genMom->status();
            simInfo[i].motherVertex = genMom->vertex();
          }
          dumpFormatedInfo(simInfo[i]);
          LogTrace("MuonSimClassifier") << "\t   has GEN mother pdgId = " << simInfo[i].motherPdgId
                                        << " (status = " << simInfo[i].motherStatus << ")";

          reco::GenParticleRef genGMom = genMom->numberOfMothers() > 0 ? genMom->motherRef() : reco::GenParticleRef();

          if (genGMom.isNonnull()) {
            simInfo[i].grandMotherPdgId = genGMom->pdgId();
            LogTrace("MuonSimClassifier")
                << "\t\t mother prod. vertex rho = " << simInfo[i].motherVertex.Rho()
                << ", z = " << simInfo[i].motherVertex.Z() << ", grand-mom pdgId = " << simInfo[i].grandMotherPdgId;
          }
          // in this case, we might want to know the heaviest mom flavour
          for (reco::GenParticleRef nMom = genMom;
               nMom.isNonnull() && std::abs(nMom->pdgId()) >= 100;  // stop when we're no longer
                                                                    // looking at hadrons or mesons
               nMom = nMom->numberOfMothers() > 0 ? nMom->motherRef() : reco::GenParticleRef()) {
            int flav = flavour(nMom->pdgId());
            if (simInfo[i].heaviestMotherFlavour < flav)
              simInfo[i].heaviestMotherFlavour = flav;
            LogTrace("MuonSimClassifier")
                << "\t\t backtracking flavour: mom pdgId = " << nMom->pdgId() << ", flavour = " << flav
                << ", heaviest so far = " << simInfo[i].heaviestMotherFlavour;
          }
        } else {  // mother is null ??
          dumpFormatedInfo(simInfo[i]);
          LogTrace("MuonSimClassifier") << "\t   has NO mother!";
        }
      } else {  // Muon is in SIM Only
        TrackingParticleRef simMom = getTpMother(tp);
        if (simMom.isNonnull()) {
          simInfo[i].motherPdgId = simMom->pdgId();
          simInfo[i].motherVertex = simMom->vertex();
          dumpFormatedInfo(simInfo[i]);
          LogTrace("MuonSimClassifier") << "\t   has SIM mother pdgId = " << simInfo[i].motherPdgId
                                        << " produced at rho = " << simMom->vertex().Rho()
                                        << ", z = " << simMom->vertex().Z();

          if (!simMom->genParticles().empty()) {
            simInfo[i].motherStatus = simMom->genParticles()[0]->status();
            reco::GenParticleRef genGMom =
                (simMom->genParticles()[0]->numberOfMothers() > 0 ? simMom->genParticles()[0]->motherRef()
                                                                  : reco::GenParticleRef());
            if (genGMom.isNonnull())
              simInfo[i].grandMotherPdgId = genGMom->pdgId();
            LogTrace("MuonSimClassifier") << "\t\t SIM mother is in GEN (status " << simInfo[i].motherStatus
                                          << "), grand-mom id = " << simInfo[i].grandMotherPdgId;
          } else {
            simInfo[i].motherStatus = -1;
            TrackingParticleRef simGMom = getTpMother(simMom);
            if (simGMom.isNonnull())
              simInfo[i].grandMotherPdgId = simGMom->pdgId();
            LogTrace("MuonSimClassifier")
                << "\t\t SIM mother is in SIM only, grand-mom id = " << simInfo[i].grandMotherPdgId;
          }
        } else {
          dumpFormatedInfo(simInfo[i]);
          LogTrace("MuonSimClassifier") << "\t   has NO mother!";
        }
      }
      simInfo[i].motherFlavour = flavour(simInfo[i].motherPdgId);
      simInfo[i].grandMotherFlavour = flavour(simInfo[i].grandMotherPdgId);

      // Check first IF this is a muon at all
      if (std::abs(tp->pdgId()) != 13) {
        if (std::abs(tp->pdgId()) == 11) {
          simInfo[i].primaryClass = isGhost ? reco::MuonSimType::GhostElectron : reco::MuonSimType::MatchedElectron;
          simInfo[i].extendedClass =
              isGhost ? reco::ExtendedMuonSimType::ExtGhostElectron : reco::ExtendedMuonSimType::ExtMatchedElectron;
          LogTrace("MuonSimClassifier") << "\t This is electron/positron. classif[i] = " << simInfo[i].primaryClass;
        } else {
          simInfo[i].primaryClass =
              isGhost ? reco::MuonSimType::GhostPunchthrough : reco::MuonSimType::MatchedPunchthrough;
          simInfo[i].extendedClass = isGhost ? reco::ExtendedMuonSimType::ExtGhostPunchthrough
                                             : reco::ExtendedMuonSimType::ExtMatchedPunchthrough;
          LogTrace("MuonSimClassifier") << "\t This is not a muon. Sorry. classif[i] = " << simInfo[i].primaryClass;
        }
        continue;
      }

      // Is this SIM muon also a GEN muon, with a mother?
      if (!tp->genParticles().empty() && (simInfo[i].motherPdgId != 0)) {
        if (std::abs(simInfo[i].motherPdgId) < 100 && (std::abs(simInfo[i].motherPdgId) != 15)) {
          simInfo[i].primaryClass =
              isGhost ? reco::MuonSimType::GhostPrimaryMuon : reco::MuonSimType::MatchedPrimaryMuon;
          simInfo[i].flavour = 13;
          simInfo[i].extendedClass = isGhost ? reco::ExtendedMuonSimType::GhostMuonFromGaugeOrHiggsBoson
                                             : reco::ExtendedMuonSimType::MatchedMuonFromGaugeOrHiggsBoson;
          LogTrace("MuonSimClassifier") << "\t This seems PRIMARY MUON ! classif[i] = " << simInfo[i].primaryClass;
        } else if (simInfo[i].motherFlavour == 4 || simInfo[i].motherFlavour == 5 || simInfo[i].motherFlavour == 15) {
          simInfo[i].primaryClass =
              isGhost ? reco::MuonSimType::GhostMuonFromHeavyFlavour : reco::MuonSimType::MatchedMuonFromHeavyFlavour;
          simInfo[i].flavour = simInfo[i].motherFlavour;
          if (simInfo[i].motherFlavour == 15)
            simInfo[i].extendedClass =
                isGhost ? reco::ExtendedMuonSimType::GhostMuonFromTau : reco::ExtendedMuonSimType::MatchedMuonFromTau;
          else if (simInfo[i].motherFlavour == 5)
            simInfo[i].extendedClass =
                isGhost ? reco::ExtendedMuonSimType::GhostMuonFromB : reco::ExtendedMuonSimType::MatchedMuonFromB;
          else if (simInfo[i].heaviestMotherFlavour == 5)
            simInfo[i].extendedClass =
                isGhost ? reco::ExtendedMuonSimType::GhostMuonFromBtoC : reco::ExtendedMuonSimType::MatchedMuonFromBtoC;
          else
            simInfo[i].extendedClass =
                isGhost ? reco::ExtendedMuonSimType::GhostMuonFromC : reco::ExtendedMuonSimType::MatchedMuonFromC;
          LogTrace("MuonSimClassifier") << "\t This seems HEAVY FLAVOUR ! classif[i] = " << simInfo[i].primaryClass;
        } else {
          simInfo[i].primaryClass =
              isGhost ? reco::MuonSimType::GhostMuonFromLightFlavour : reco::MuonSimType::MatchedMuonFromLightFlavour;
          simInfo[i].flavour = simInfo[i].motherFlavour;
          LogTrace("MuonSimClassifier") << "\t This seems LIGHT FLAVOUR ! classif[i] = " << simInfo[i].primaryClass;
        }
      } else {
        simInfo[i].primaryClass =
            isGhost ? reco::MuonSimType::GhostMuonFromLightFlavour : reco::MuonSimType::MatchedMuonFromLightFlavour;
        simInfo[i].flavour = simInfo[i].motherFlavour;
        LogTrace("MuonSimClassifier") << "\t This seems LIGHT FLAVOUR ! classif[i] = " << simInfo[i].primaryClass;
      }

      // extended classification
      // don't we override previous decisions?
      if (simInfo[i].motherPdgId == 0)
        // if it has no mom, it's not a primary particle so it won't be in ppMuX
        simInfo[i].extendedClass = isGhost ? reco::ExtendedMuonSimType::GhostMuonFromNonPrimaryParticle
                                           : reco::ExtendedMuonSimType::MatchedMuonFromNonPrimaryParticle;
      else if (std::abs(simInfo[i].motherPdgId) < 100) {
        if (simInfo[i].motherFlavour == 15)
          simInfo[i].extendedClass =
              isGhost ? reco::ExtendedMuonSimType::GhostMuonFromTau : reco::ExtendedMuonSimType::MatchedMuonFromTau;
        else
          simInfo[i].extendedClass = isGhost ? reco::ExtendedMuonSimType::GhostMuonFromGaugeOrHiggsBoson
                                             : reco::ExtendedMuonSimType::MatchedMuonFromGaugeOrHiggsBoson;
      } else if (simInfo[i].motherFlavour == 5)
        simInfo[i].extendedClass =
            isGhost ? reco::ExtendedMuonSimType::GhostMuonFromB : reco::ExtendedMuonSimType::MatchedMuonFromB;
      else if (simInfo[i].motherFlavour == 4) {
        if (simInfo[i].heaviestMotherFlavour == 5)
          simInfo[i].extendedClass =
              isGhost ? reco::ExtendedMuonSimType::GhostMuonFromBtoC : reco::ExtendedMuonSimType::MatchedMuonFromBtoC;
        else
          simInfo[i].extendedClass =
              isGhost ? reco::ExtendedMuonSimType::GhostMuonFromC : reco::ExtendedMuonSimType::MatchedMuonFromC;
      } else if (simInfo[i].motherStatus != -1) {  // primary light particle
        int id = std::abs(simInfo[i].motherPdgId);
        if (id != /*pi+*/ 211 && id != /*K+*/ 321 && id != 130 /*K0L*/)
          // other light particle, possibly short-lived
          simInfo[i].extendedClass = isGhost ? reco::ExtendedMuonSimType::GhostMuonFromOtherLight
                                             : reco::ExtendedMuonSimType::MatchedMuonFromOtherLight;
        else if (simInfo[i].vertex.Rho() < decayRho_ && std::abs(simInfo[i].vertex.Z()) < decayAbsZ_)
          // decay a la ppMuX (primary pi/K within a cylinder)
          simInfo[i].extendedClass = isGhost ? reco::ExtendedMuonSimType::GhostMuonFromPiKppMuX
                                             : reco::ExtendedMuonSimType::MatchedMuonFromPiKppMuX;
        else
          // late decay that wouldn't be in ppMuX
          simInfo[i].extendedClass = isGhost ? reco::ExtendedMuonSimType::GhostMuonFromPiKNotppMuX
                                             : reco::ExtendedMuonSimType::MatchedMuonFromPiKNotppMuX;
      } else
        // decay of non-primary particle, would not be in ppMuX
        simInfo[i].extendedClass = isGhost ? reco::ExtendedMuonSimType::GhostMuonFromNonPrimaryParticle
                                           : reco::ExtendedMuonSimType::MatchedMuonFromNonPrimaryParticle;

      if (linkToGenParticles_ && std::abs(simInfo[i].extendedClass) >= 2) {
        // Link to the genParticle if possible, but not decays in flight (in
        // ppMuX they're in GEN block, but they have wrong parameters)
        if (!tp->genParticles().empty() && std::abs(simInfo[i].extendedClass) >= 5) {
          if (genParticles.id() != tp->genParticles().id()) {
            throw cms::Exception("Configuration")
                << "Product ID mismatch between the genParticle collection (" << genParticles_ << ", id "
                << genParticles.id() << ") and the references in the TrackingParticles (id " << tp->genParticles().id()
                << ")\n";
          }
          muToPrimary[i] = tp->genParticles()[0].key();
        } else {
          // Don't put the same trackingParticle twice!
          int &indexPlus1 = tpToSecondaries[tp];  // will create a 0 if the tp is
                                                  // not in the list already
          if (indexPlus1 == 0)
            indexPlus1 = convertAndPush(*tp, *secondaries, getTpMother(tp), genParticles) + 1;
          muToSecondary[i] = indexPlus1 - 1;
        }
      }
      LogTrace("MuonSimClassifier") << "\t Extended classification code = " << simInfo[i].extendedClass;
    } else {  // if (tp.isNonnull())
      simInfo[i].primaryClass = reco::MuonSimType::NotMatched;
      simInfo[i].extendedClass = reco::ExtendedMuonSimType::ExtNotMatched;
    }
  }  // end loop on reco muons

  writeValueMap(iEvent, muons, simInfo, "");

  if (linkToGenParticles_) {
    edm::OrphanHandle<reco::GenParticleCollection> secHandle = iEvent.put(std::move(secondaries), "secondaries");
    edm::RefProd<reco::GenParticleCollection> priRP(genParticles);
    edm::RefProd<reco::GenParticleCollection> secRP(secHandle);
    std::unique_ptr<edm::Association<reco::GenParticleCollection>> outPri(
        new edm::Association<reco::GenParticleCollection>(priRP));
    std::unique_ptr<edm::Association<reco::GenParticleCollection>> outSec(
        new edm::Association<reco::GenParticleCollection>(secRP));
    edm::Association<reco::GenParticleCollection>::Filler fillPri(*outPri), fillSec(*outSec);
    fillPri.insert(muons, muToPrimary.begin(), muToPrimary.end());
    fillSec.insert(muons, muToSecondary.begin(), muToSecondary.end());
    fillPri.fill();
    fillSec.fill();
    iEvent.put(std::move(outPri), "toPrimaries");
    iEvent.put(std::move(outSec), "toSecondaries");
  }
}

template <typename T>
void MuonSimClassifier::writeValueMap(edm::Event &iEvent,
                                      const edm::Handle<edm::View<reco::Muon>> &handle,
                                      const std::vector<T> &values,
                                      const std::string &label) const {
  using namespace edm;
  using namespace std;
  unique_ptr<ValueMap<T>> valMap(new ValueMap<T>());
  typename edm::ValueMap<T>::Filler filler(*valMap);
  filler.insert(handle, values.begin(), values.end());
  filler.fill();
  iEvent.put(std::move(valMap), label);
}

int MuonSimClassifier::flavour(int pdgId) const {
  int flav = std::abs(pdgId);
  // for quarks, leptons and bosons except gluons, take their pdgId
  // muons and taus have themselves as flavour
  if (flav <= 37 && flav != 21)
    return flav;
  // look for barions
  int bflav = ((flav / 1000) % 10);
  if (bflav != 0)
    return bflav;
  // look for mesons
  int mflav = ((flav / 100) % 10);
  if (mflav != 0)
    return mflav;
  return 0;
}

// push secondary in collection.
// if it has a primary mother link to it.
int MuonSimClassifier::convertAndPush(const TrackingParticle &tp,
                                      reco::GenParticleCollection &out,
                                      const TrackingParticleRef &simMom,
                                      const edm::Handle<reco::GenParticleCollection> &genParticles) const {
  out.push_back(reco::GenParticle(tp.charge(), tp.p4(), tp.vertex(), tp.pdgId(), tp.status(), true));
  if (simMom.isNonnull() && !simMom->genParticles().empty()) {
    if (genParticles.id() != simMom->genParticles().id()) {
      throw cms::Exception("Configuration")
          << "Product ID mismatch between the genParticle collection (" << genParticles_ << ", id " << genParticles.id()
          << ") and the references in the TrackingParticles (id " << simMom->genParticles().id() << ")\n";
    }
    out.back().addMother(simMom->genParticles()[0]);
  }
  return out.size() - 1;
}

// define this as a plug-in
DEFINE_FWK_MODULE(MuonSimClassifier);
