// -*- C++ -*-
//
// Package:    V0Validator
// Class:      V0Validator
//
/**\class V0Validator V0Validator.cc Validation/RecoVertex/src/V0Validator.cc

 Description: Creates validation histograms for RecoVertex/V0Producer

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Wed Feb 18 17:21:04 MST 2009
//
//

#include "Validation/RecoVertex/interface/V0Validator.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

typedef std::vector<TrackingVertex> TrackingVertexCollection;
typedef edm::Ref<TrackingVertexCollection> TrackingVertexRef;
typedef edm::RefVector<edm::HepMCProduct, HepMC::GenVertex> GenVertexRefVector;
typedef edm::RefVector<edm::HepMCProduct, HepMC::GenParticle> GenParticleRefVector;

V0Validator::V0Validator(const edm::ParameterSet& iConfig)
    : theDQMRootFileName(iConfig.getUntrackedParameter<std::string>("DQMRootFileName")),
      dirName(iConfig.getUntrackedParameter<std::string>("dirName")),
      recoRecoToSimCollectionToken_(
          consumes<reco::RecoToSimCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociatorMap"))),
      recoSimToRecoCollectionToken_(
          consumes<reco::SimToRecoCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociatorMap"))),
      trackingVertexCollection_Token_(
          consumes<TrackingVertexCollection>(iConfig.getUntrackedParameter<edm::InputTag>("trackingVertexCollection"))),
      vec_recoVertex_Token_(
          consumes<std::vector<reco::Vertex> >(iConfig.getUntrackedParameter<edm::InputTag>("vertexCollection"))),
      recoVertexCompositeCandidateCollection_k0s_Token_(consumes<reco::VertexCompositeCandidateCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("kShortCollection"))),
      recoVertexCompositeCandidateCollection_lambda_Token_(consumes<reco::VertexCompositeCandidateCollection>(
          iConfig.getUntrackedParameter<edm::InputTag>("lambdaCollection"))) {}

V0Validator::~V0Validator() {}

void V0Validator::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) {
  double minKsMass = 0.49767 - 0.07;
  double maxKsMass = 0.49767 + 0.07;
  double minLamMass = 1.1156 - 0.05;
  double maxLamMass = 1.1156 + 0.05;
  int ksMassNbins = 100;
  double ksMassXmin = minKsMass;
  double ksMassXmax = maxKsMass;
  int lamMassNbins = 100;
  double lamMassXmin = minLamMass;
  double lamMassXmax = maxLamMass;

  ibooker.cd();
  std::string subDirName = V0Validator::dirName + "/K0";
  ibooker.setCurrentFolder(subDirName);

  candidateEffVsR_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sEffVsR_num", "K^{0}_{S} Efficiency vs #rho", 80, 0., 40.);
  candidateEffVsEta_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sEffVsEta_num", "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  candidateEffVsPt_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sEffVsPt_num", "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);

  candidateTkEffVsR_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sTkEffVsR_num", "K^{0}_{S} Tracking Efficiency vs #rho", 80, 0., 40.);
  candidateTkEffVsEta_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sTkEffVsEta_num", "K^{0}_{S} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  candidateTkEffVsPt_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sTkEffVsPt_num", "K^{0}_{S} Tracking Efficiency vs p_{T}", 70, 0., 20.);

  candidateEffVsR_denom_[V0Validator::KSHORT] =
      ibooker.book1D("K0sEffVsR_denom", "K^{0}_{S} Efficiency vs #rho", 80, 0., 40.);
  candidateEffVsEta_denom_[V0Validator::KSHORT] =
      ibooker.book1D("K0sEffVsEta_denom", "K^{0}_{S} Efficiency vs #eta", 40, -2.5, 2.5);
  candidateEffVsPt_denom_[V0Validator::KSHORT] =
      ibooker.book1D("K0sEffVsPt_denom", "K^{0}_{S} Efficiency vs p_{T}", 70, 0., 20.);

  candidateFakeVsR_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sFakeVsR_num", "K^{0}_{S} Fake Rate vs #rho", 80, 0., 40.);
  candidateFakeVsEta_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sFakeVsEta_num", "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  candidateFakeVsPt_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sFakeVsPt_num", "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);
  candidateTkFakeVsR_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sTkFakeVsR_num", "K^{0}_{S} Tracking Fake Rate vs #rho", 80, 0., 80.);
  candidateTkFakeVsEta_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sTkFakeVsEta_num", "K^{0}_{S} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  candidateTkFakeVsPt_num_[V0Validator::KSHORT] =
      ibooker.book1D("K0sTkFakeVsPt_num", "K^{0}_{S} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  candidateFakeVsR_denom_[V0Validator::KSHORT] =
      ibooker.book1D("K0sFakeVsR_denom", "K^{0}_{S} Fake Rate vs #rho", 80, 0., 40.);
  candidateFakeVsEta_denom_[V0Validator::KSHORT] =
      ibooker.book1D("K0sFakeVsEta_denom", "K^{0}_{S} Fake Rate vs #eta", 40, -2.5, 2.5);
  candidateFakeVsPt_denom_[V0Validator::KSHORT] =
      ibooker.book1D("K0sFakeVsPt_denom", "K^{0}_{S} Fake Rate vs p_{T}", 70, 0., 20.);
  nCandidates_[V0Validator::KSHORT] = ibooker.book1D("nK0s", "Number of K^{0}_{S} found per event", 60, 0., 60.);
  fakeCandidateMass_[V0Validator::KSHORT] =
      ibooker.book1D("ksMassFake", "Mass of fake K0S", ksMassNbins, minKsMass, maxKsMass);
  goodCandidateMass[V0Validator::KSHORT] =
      ibooker.book1D("ksMassGood", "Mass of good reco K0S", ksMassNbins, minKsMass, maxKsMass);
  candidateMassAll[V0Validator::KSHORT] =
      ibooker.book1D("ksMassAll", "Invariant mass of all K0S", ksMassNbins, ksMassXmin, ksMassXmax);
  candidateFakeDauRadDist_[V0Validator::KSHORT] =
      ibooker.book1D("radDistFakeKs", "Production radius of daughter particle of Ks fake", 100, 0., 15.);
  candidateStatus_[V0Validator::KSHORT] = ibooker.book1D("ksCandStatus", "Fake type by cand status", 10, 0., 10.);

  // Lambda Plots follow

  subDirName = V0Validator::dirName + "/Lambda";
  ibooker.setCurrentFolder(subDirName);

  candidateEffVsR_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamEffVsR_num", "#Lambda^{0} Efficiency vs #rho", 80, 0., 40.);
  candidateEffVsEta_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamEffVsEta_num", "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  candidateEffVsPt_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamEffVsPt_num", "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);

  candidateTkEffVsR_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamTkEffVsR_num", "#Lambda^{0} TrackingEfficiency vs #rho", 80, 0., 40.);
  candidateTkEffVsEta_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamTkEffVsEta_num", "#Lambda^{0} Tracking Efficiency vs #eta", 40, -2.5, 2.5);
  candidateTkEffVsPt_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamTkEffVsPt_num", "#Lambda^{0} Tracking Efficiency vs p_{T}", 70, 0., 20.);

  candidateEffVsR_denom_[V0Validator::LAMBDA] =
      ibooker.book1D("LamEffVsR_denom", "#Lambda^{0} Efficiency vs #rho", 80, 0., 40.);
  candidateEffVsEta_denom_[V0Validator::LAMBDA] =
      ibooker.book1D("LamEffVsEta_denom", "#Lambda^{0} Efficiency vs #eta", 40, -2.5, 2.5);
  candidateEffVsPt_denom_[V0Validator::LAMBDA] =
      ibooker.book1D("LamEffVsPt_denom", "#Lambda^{0} Efficiency vs p_{T}", 70, 0., 20.);

  candidateFakeVsR_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamFakeVsR_num", "#Lambda^{0} Fake Rate vs #rho", 80, 0., 40.);
  candidateFakeVsEta_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamFakeVsEta_num", "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  candidateFakeVsPt_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamFakeVsPt_num", "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);
  candidateTkFakeVsR_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamTkFakeVsR_num", "#Lambda^{0} Tracking Fake Rate vs #rho", 80, 0., 40.);
  candidateTkFakeVsEta_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamTkFakeVsEta_num", "#Lambda^{0} Tracking Fake Rate vs #eta", 40, -2.5, 2.5);
  candidateTkFakeVsPt_num_[V0Validator::LAMBDA] =
      ibooker.book1D("LamTkFakeVsPt_num", "#Lambda^{0} Tracking Fake Rate vs p_{T}", 70, 0., 20.);

  candidateFakeVsR_denom_[V0Validator::LAMBDA] =
      ibooker.book1D("LamFakeVsR_denom", "#Lambda^{0} Fake Rate vs #rho", 80, 0., 40.);
  candidateFakeVsEta_denom_[V0Validator::LAMBDA] =
      ibooker.book1D("LamFakeVsEta_denom", "#Lambda^{0} Fake Rate vs #eta", 40, -2.5, 2.5);
  candidateFakeVsPt_denom_[V0Validator::LAMBDA] =
      ibooker.book1D("LamFakeVsPt_denom", "#Lambda^{0} Fake Rate vs p_{T}", 70, 0., 20.);

  nCandidates_[V0Validator::LAMBDA] = ibooker.book1D("nLam", "Number of #Lambda^{0} found per event", 60, 0., 60.);
  fakeCandidateMass_[V0Validator::LAMBDA] =
      ibooker.book1D("lamMassFake", "Mass of fake Lambda", lamMassNbins, minLamMass, maxLamMass);
  goodCandidateMass[V0Validator::LAMBDA] =
      ibooker.book1D("lamMassGood", "Mass of good Lambda", lamMassNbins, minLamMass, maxLamMass);

  candidateMassAll[V0Validator::LAMBDA] =
      ibooker.book1D("lamMassAll", "Invariant mass of all #Lambda^{0}", lamMassNbins, lamMassXmin, lamMassXmax);
  candidateFakeDauRadDist_[V0Validator::LAMBDA] =
      ibooker.book1D("radDistFakeLam", "Production radius of daughter particle of Lam fake", 100, 0., 15.);

  candidateStatus_[V0Validator::LAMBDA] = ibooker.book1D("ksCandStatus", "Fake type by cand status", 10, 0., 10.);
}

void V0Validator::doFakeRates(const reco::VertexCompositeCandidateCollection& collection,
                              const reco::RecoToSimCollection& recotosimCollection,
                              V0Type v0_type,
                              int particle_pdgid,
                              int misreconstructed_particle_pdgid) {
  using namespace edm;

  int numCandidateFound = 0;
  double mass = 0.;
  float CandidatepT = 0.;
  float CandidateEta = 0.;
  float CandidateR = 0.;
  int CandidateStatus = 0;
  const unsigned int NUM_DAUGHTERS = 2;
  if (!collection.empty()) {
    for (reco::VertexCompositeCandidateCollection::const_iterator iCandidate = collection.begin();
         iCandidate != collection.end();
         iCandidate++) {
      // Fill values to be histogrammed
      mass = iCandidate->mass();
      CandidatepT = (sqrt(iCandidate->momentum().perp2()));
      CandidateEta = iCandidate->momentum().eta();
      CandidateR = (sqrt(iCandidate->vertex().perp2()));
      candidateMassAll[v0_type]->Fill(mass);
      CandidateStatus = 0;

      std::array<reco::TrackRef, NUM_DAUGHTERS> theDaughterTracks = {
          {(*(dynamic_cast<const reco::RecoChargedCandidate*>(iCandidate->daughter(0)))).track(),
           (*(dynamic_cast<const reco::RecoChargedCandidate*>(iCandidate->daughter(1)))).track()}};

      TrackingParticleRef tpref;
      TrackingParticleRef firstDauTP;
      TrackingVertexRef candidateVtx;

      std::array<double, NUM_DAUGHTERS> radDist;
      // Loop through candidate's daugher tracks
      for (View<reco::Track>::size_type i = 0; i < theDaughterTracks.size(); ++i) {
        radDist = {{-1., -1.}};
        // Found track from theDaughterTracks
        RefToBase<reco::Track> track(theDaughterTracks.at(i));

        if (recotosimCollection.find(track) != recotosimCollection.end()) {
          const std::vector<std::pair<TrackingParticleRef, double> >& tp = recotosimCollection[track];
          if (!tp.empty()) {
            tpref = tp.begin()->first;

            TrackingVertexRef parentVertex = tpref->parentVertex();
            if (parentVertex.isNonnull()) {
              radDist[i] = parentVertex->position().R();
              if (candidateVtx.isNonnull()) {
                if (candidateVtx->position() == parentVertex->position()) {
                  if (parentVertex->nDaughterTracks() == 2) {
                    if (parentVertex->nSourceTracks() == 0) {
                      // No source tracks found for candidate's
                      // vertex: it shouldn't happen, but does for
                      // evtGen events
                      CandidateStatus = 6;
                    }

                    for (TrackingVertex::tp_iterator iTP = parentVertex->sourceTracks_begin();
                         iTP != parentVertex->sourceTracks_end();
                         iTP++) {
                      if (abs((*iTP)->pdgId()) == particle_pdgid) {
                        CandidateStatus = 1;
                        numCandidateFound += 1.;
                        goodCandidateMass[v0_type]->Fill(mass);
                      } else {
                        CandidateStatus = 2;
                        if (abs((*iTP)->pdgId()) == misreconstructed_particle_pdgid) {
                          CandidateStatus = 7;
                        }
                      }
                    }
                  } else {
                    // Found a bad match because the mother has too
                    // many daughters
                    CandidateStatus = 3;
                  }
                } else {
                  // Found a bad match because the parent vertices
                  // from the two tracks are different
                  CandidateStatus = 4;
                }
              } else {
                // if candidateVtx is null, fill it with parentVertex
                // to compare to the parentVertex from the second
                // track
                candidateVtx = parentVertex;
                firstDauTP = tpref;
              }
            }  // parent vertex is null
          }    // check on associated tp size zero
        } else {
          CandidateStatus = 5;
        }
      }  // Loop on candidate's daughter tracks

      // fill the fake rate histograms
      if (CandidateStatus > 1) {
        candidateFakeVsR_num_[v0_type]->Fill(CandidateR);
        candidateFakeVsEta_num_[v0_type]->Fill(CandidateEta);
        candidateFakeVsPt_num_[v0_type]->Fill(CandidatepT);
        candidateStatus_[v0_type]->Fill((float)CandidateStatus);
        fakeCandidateMass_[v0_type]->Fill(mass);
        for (auto distance : radDist) {
          if (distance > 0)
            candidateFakeDauRadDist_[v0_type]->Fill(distance);
        }
      }
      if (CandidateStatus == 5) {
        candidateTkFakeVsR_num_[v0_type]->Fill(CandidateR);
        candidateTkFakeVsEta_num_[v0_type]->Fill(CandidateEta);
        candidateTkFakeVsPt_num_[v0_type]->Fill(CandidatepT);
      }
      candidateFakeVsR_denom_[v0_type]->Fill(CandidateR);
      candidateFakeVsEta_denom_[v0_type]->Fill(CandidateEta);
      candidateFakeVsPt_denom_[v0_type]->Fill(CandidatepT);
    }  // Loop on candidates
  }    // check on presence of candidate's collection in the event
  nCandidates_[v0_type]->Fill((float)numCandidateFound);
}

void V0Validator::doEfficiencies(const TrackingVertexCollection& gen_vertices,
                                 V0Type v0_type,
                                 int parent_particle_id,
                                 int first_daughter_id,  /* give only positive charge */
                                 int second_daughter_id, /* give only positive charge */
                                 const reco::VertexCompositeCandidateCollection& collection,
                                 const reco::SimToRecoCollection& simtorecoCollection) {
  /* We store the TrackRef of the tracks that have been used to
   * produce the V0 under consideration here. This is used later to
   * check if a specific V0 has been really reconstructed or not. The
   * ordering is based on the key_index of the reference, since it
   * indeed does not matter that much. */

  std::set<V0Couple> reconstructed_V0_couples;
  if (!collection.empty()) {
    for (reco::VertexCompositeCandidateCollection::const_iterator iCandidate = collection.begin();
         iCandidate != collection.end();
         iCandidate++) {
      reconstructed_V0_couples.insert(
          V0Couple((dynamic_cast<const reco::RecoChargedCandidate*>(iCandidate->daughter(0)))->track(),
                   (dynamic_cast<const reco::RecoChargedCandidate*>(iCandidate->daughter(1)))->track()));
    }
  }

  /* PSEUDO CODE
     for v in gen_vertices
       if v.eventId().BX() !=0 continue
       if v.nDaughterTracks != 2 continue
       for source in v.sourceTracks_begin
         if source is parent_particle_id
          for daughter in v.daughterTracks_begin
           if daughter in region_and_kine_cuts
             decay_found
   */
  unsigned int candidateEff[2] = {0, 0};
  for (auto const& gen_vertex : gen_vertices) {
    if (gen_vertex.eventId().bunchCrossing() != 0)
      continue;  // Consider only in-time events
    if (gen_vertex.nDaughterTracks() != 2)
      continue;  // Keep only V0 vertices
    for (TrackingVertex::tp_iterator source = gen_vertex.sourceTracks_begin(); source != gen_vertex.sourceTracks_end();
         ++source) {
      if (std::abs((*source)->pdgId()) == parent_particle_id) {
        if ((std::abs((gen_vertex.daughterTracks().at(0))->pdgId()) == first_daughter_id &&
             std::abs((gen_vertex.daughterTracks().at(1))->pdgId()) == second_daughter_id) ||
            (std::abs((gen_vertex.daughterTracks().at(0))->pdgId()) == second_daughter_id &&
             std::abs((gen_vertex.daughterTracks().at(1))->pdgId()) == first_daughter_id)) {
          if ((std::abs((gen_vertex.daughterTracks().at(0))->momentum().eta()) < 2.4 &&
               gen_vertex.daughterTracks().at(0)->pt() > 0.9) &&
              (std::abs((gen_vertex.daughterTracks().at(1))->momentum().eta()) < 2.4 &&
               gen_vertex.daughterTracks().at(1)->pt() > 0.9)) {
            // found desired generated Candidate
            float candidateGenpT = sqrt((*source)->momentum().perp2());
            float candidateGenEta = (*source)->momentum().eta();
            float candidateGenR = sqrt((*source)->vertex().perp2());
            candidateEffVsPt_denom_[v0_type]->Fill(candidateGenpT);
            candidateEffVsEta_denom_[v0_type]->Fill(candidateGenEta);
            candidateEffVsR_denom_[v0_type]->Fill(candidateGenR);

            std::array<reco::TrackRef, 2> reco_daughter;

            for (unsigned int daughter = 0; daughter < 2; ++daughter) {
              if (simtorecoCollection.find(gen_vertex.daughterTracks()[daughter]) != simtorecoCollection.end()) {
                if (!simtorecoCollection[gen_vertex.daughterTracks()[daughter]].empty()) {
                  candidateEff[daughter] = 1;  // Found a daughter track
                  reco_daughter[daughter] =
                      simtorecoCollection[gen_vertex.daughterTracks()[daughter]].begin()->first.castTo<reco::TrackRef>();
                }
              } else {
                candidateEff[daughter] = 2;  // First daughter not found
              }
            }
            if ((candidateEff[0] == 1 && candidateEff[1] == 1) && (reco_daughter[0].key() != reco_daughter[1].key()) &&
                (reconstructed_V0_couples.find(V0Couple(reco_daughter[0], reco_daughter[1])) !=
                 reconstructed_V0_couples.end())) {
              candidateEffVsPt_num_[v0_type]->Fill(candidateGenpT);
              candidateEffVsEta_num_[v0_type]->Fill(candidateGenEta);
              candidateEffVsR_num_[v0_type]->Fill(candidateGenR);
            }
          }  // Check that daughters are inside the desired kinematic region
        }    // Check decay products of the current generatex vertex
      }      // Check pdgId of the source of the current generated vertex
    }        // Loop over all sources of the current generated vertex
  }          // Loop over all generated vertices
}

void V0Validator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using std::cout;
  using std::endl;
  using namespace edm;
  using namespace std;

  // Make matching collections
  Handle<reco::RecoToSimCollection> recotosimCollectionH;
  iEvent.getByToken(recoRecoToSimCollectionToken_, recotosimCollectionH);

  Handle<reco::SimToRecoCollection> simtorecoCollectionH;
  iEvent.getByToken(recoSimToRecoCollectionToken_, simtorecoCollectionH);

  // Get Monte Carlo information
  edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByToken(trackingVertexCollection_Token_, TVCollectionH);

  // Select the primary vertex, create a new reco::Vertex to hold it
  edm::Handle<std::vector<reco::Vertex> > primaryVtxCollectionH;
  iEvent.getByToken(vec_recoVertex_Token_, primaryVtxCollectionH);

  std::vector<reco::Vertex>::const_iterator iVtxPH = primaryVtxCollectionH->begin();
  for (std::vector<reco::Vertex>::const_iterator iVtx = primaryVtxCollectionH->begin();
       iVtx < primaryVtxCollectionH->end();
       iVtx++) {
    if (primaryVtxCollectionH->size() > 1) {
      if (iVtx->tracksSize() > iVtxPH->tracksSize()) {
        iVtxPH = iVtx;
      }
    } else
      iVtxPH = iVtx;
  }

  // get the V0s;
  edm::Handle<reco::VertexCompositeCandidateCollection> k0sCollection;
  edm::Handle<reco::VertexCompositeCandidateCollection> lambdaCollection;
  iEvent.getByToken(recoVertexCompositeCandidateCollection_k0s_Token_, k0sCollection);
  iEvent.getByToken(recoVertexCompositeCandidateCollection_lambda_Token_, lambdaCollection);

  // Do fake rate and efficiency calculation

  // Get gen vertex collection out of the event, as done in the Vertex
  // validation package!!!
  if (k0sCollection.isValid()) {
    doFakeRates(*k0sCollection.product(), *recotosimCollectionH.product(), V0Type::KSHORT, 310, 3122);
    doEfficiencies(*TVCollectionH.product(),
                   V0Type::KSHORT,
                   310,
                   211,
                   211,
                   *k0sCollection.product(),
                   *simtorecoCollectionH.product());
  }
  if (lambdaCollection.isValid()) {
    doFakeRates(*lambdaCollection.product(), *recotosimCollectionH.product(), V0Type::LAMBDA, 3122, 310);
    doEfficiencies(*TVCollectionH.product(),
                   V0Type::LAMBDA,
                   3122,
                   211,
                   2212,
                   *lambdaCollection.product(),
                   *simtorecoCollectionH.product());
  }
}

// define this as a plug-in
// DEFINE_FWK_MODULE(V0Validator);
