#include "SimG4Core/CustomPhysics/interface/RHDecayTracer.h"
#include "SimG4Core/CustomPhysics/interface/RHadronPythiaDecayDataManager.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include <tuple>

// Producer that adds R-hadron history and decay information from RHadronPythiaDecayer to a GenParticleCollection
using TrackData = RHadronPythiaDecayDataManager::TrackData;

RHDecayTracer::RHDecayTracer(edm::ParameterSet const& p) : edm::stream::EDProducer<>() {
  edmSimTrackContainerToken_ =
      consumes<edm::SimTrackContainer>(p.getUntrackedParameter<edm::InputTag>("G4TrkSrc", edm::InputTag("g4SimHits")));
  edmSimVertexContainerToken_ =
      consumes<edm::SimVertexContainer>(p.getUntrackedParameter<edm::InputTag>("G4VtxSrc", edm::InputTag("g4SimHits")));
  genParticleToken_ = consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"));
  genParticleRHadronDecayToken_ = produces<reco::GenParticleCollection>("RHadronDecay");
  std::vector<std::string> acceptedPDGs = p.getUntrackedParameter<std::vector<std::string>>(
      "RHDecayTracerPDGs", std::vector<std::string>{"1000600-1999999", "1000021", "1000006"});
  updateConfiguredPDGs(acceptedPDGs);
}

void RHDecayTracer::produce(edm::Event& iEvent, const edm::EventSetup&) {
  // Create the GenParticleCollection
  std::unique_ptr<reco::GenParticleCollection> genParticles = std::make_unique<reco::GenParticleCollection>();

  // Get the token to later create references
  const edm::RefProd<reco::GenParticleCollection> refProd = iEvent.getRefBeforePut(genParticleRHadronDecayToken_);

  // Add the history of the R-hadron from the GenParticle collection
  addGenRhadronHistoryToCollection(iEvent, *genParticles, refProd);

  // Add the history of the R-hadron from the SimTrack collection
  addSimRhadronHistoryToCollection(iEvent, *genParticles, refProd);

  // Add the R-hadron decay information to the collection
  addRhadronDecayToCollection(*genParticles, refProd);

  // Put the collection in the event
  iEvent.put(genParticleRHadronDecayToken_, std::move(genParticles));
}

void RHDecayTracer::addGenRhadronHistoryToCollection(edm::Event& iEvent,
                                                     reco::GenParticleCollection& genParticles,
                                                     const edm::RefProd<reco::GenParticleCollection>& refProd) {
  edm::Handle<reco::GenParticleCollection> originalGenParticles;
  iEvent.getByToken(genParticleToken_, originalGenParticles);

  // Add all R-hadrons from the GenParticle collection to the new collection
  for (const auto& originalGenParticle : *originalGenParticles) {
    int pdgId = std::abs(originalGenParticle.pdgId());
    if (isConfiguredPDG(pdgId)) {
      // Skip the daughterless R-hadrons. These will be added later in RHDecayTracer::addSimRhadronHistoryToCollection()
      if (originalGenParticle.numberOfDaughters() == 0)
        continue;
      genParticles.push_back(originalGenParticle);
    }
  }
}

void RHDecayTracer::addSimRhadronHistoryToCollection(edm::Event& iEvent,
                                                     reco::GenParticleCollection& genParticles,
                                                     const edm::RefProd<reco::GenParticleCollection>& refProd) {
  edm::Handle<edm::SimTrackContainer> simTracks;
  edm::Handle<edm::SimVertexContainer> simVertices;
  iEvent.getByToken(edmSimTrackContainerToken_, simTracks);
  iEvent.getByToken(edmSimVertexContainerToken_, simVertices);

  // Add R-hadrons from the SimTrack collection to the new collection
  for (const auto& simTrack : *simTracks) {
    int pdgId = std::abs(simTrack.type());
    if (isConfiguredPDG(pdgId)) {
      // Get the vertex associated with the SimTrack. The position of the vertex will be needed to construct the genParticle
      int vertIndex = simTrack.vertIndex();
      const SimVertex& simVertex = (*simVertices)[vertIndex];
      int parentId = simVertex.parentIndex();
      math::XYZPoint vertexPosition(simVertex.position().x(), simVertex.position().y(), simVertex.position().z());

      // Create a new gen particle from the SimTrack
      genParticles.emplace_back(simTrack.charge(), simTrack.momentum(), vertexPosition, simTrack.type(), 2, true);
      refIndexVector.push_back(genParticles.size() - 1);
      parentIdVector.push_back(parentId);
      trackIdVector.push_back(simTrack.trackId());
    }
  }

  // Add mother-daughter relationships between the R-hadrons
  linkSimTrackHistory(genParticles, refProd, refIndexVector, parentIdVector, trackIdVector);
}

void RHDecayTracer::addRhadronDecayToCollection(reco::GenParticleCollection& genParticles,
                                                const edm::RefProd<reco::GenParticleCollection>& refProd) {
  // Get track data from RHadronPythiaDecayDataManager
  std::map<int, TrackData> decayParents;
  std::map<int, std::vector<TrackData>> decayDaughters;
  gRHadronPythiaDecayDataManager->getDecayInfo(decayParents, decayDaughters);

  // Skip if no decays were recorded
  if (decayParents.empty())
    return;

  // Store parent index ranges for later linking {parentIndex, daughterStartIndex, daughterEndIndex}
  std::map<int, std::tuple<size_t, size_t, size_t>> parentRanges;

  // Loop over each decay parent and create GenParticles
  for (const auto& parentEntry : decayParents) {
    const int decayID = parentEntry.first;
    const TrackData& parentData = parentEntry.second;

    // Grab the parent R-hadron from genParticles
    size_t parentIndex = getDecayParentIndex(parentData, refIndexVector, trackIdVector);
    if (parentIndex == SIZE_MAX)
      continue;

    // Record the start index for daughters of this decay
    size_t daughterStartIndex = genParticles.size();

    // Create the daughter GenParticles
    if (decayDaughters.find(decayID) != decayDaughters.end()) {
      addDecayDaughtersToCollection(decayDaughters.at(decayID), genParticles);
    }

    size_t daughterEndIndex = genParticles.size() - 1;
    parentRanges[decayID] = {parentIndex, daughterStartIndex, daughterEndIndex};
  }

  // Add mother-daughter relationships
  for (const auto& entry : parentRanges) {
    size_t parentIndex = std::get<0>(entry.second);
    size_t daughterStartIndex = std::get<1>(entry.second);
    size_t daughterEndIndex = std::get<2>(entry.second);

    // Skip if no daughters were added
    if (daughterStartIndex > daughterEndIndex)
      continue;

    reco::GenParticleRef parentRef(refProd, parentIndex);
    reco::GenParticle& parent = (genParticles)[parentIndex];

    for (size_t i = daughterStartIndex; i <= daughterEndIndex; ++i) {
      reco::GenParticleRef daughterRef(refProd, i);
      (genParticles)[i].addMother(parentRef);
      parent.addDaughter(daughterRef);
    }
  }

  // Clear data for the next event
  gRHadronPythiaDecayDataManager->clearDecayInfo();
  refIndexVector.clear();
  parentIdVector.clear();
  trackIdVector.clear();
}

void RHDecayTracer::linkSimTrackHistory(reco::GenParticleCollection& genParticles,
                                        const edm::RefProd<reco::GenParticleCollection>& refProd,
                                        std::vector<size_t>& refIndexVector,
                                        std::vector<int>& parentIdVector,
                                        std::vector<unsigned int>& trackIdVector) {
  for (size_t i = 0; i < parentIdVector.size(); i++) {
    int parentId = parentIdVector[i];
    // Skip parentId == -1, as these have no mother and the daughter will be added shortly
    if (parentId == -1)
      continue;

    // Find the parent genParticle whose trackId is equivalent to the current parentId
    size_t parentRefIndex = SIZE_MAX;
    for (size_t j = 0; j < trackIdVector.size(); j++) {
      if (static_cast<int>(trackIdVector[j]) == parentId) {
        parentRefIndex = refIndexVector[j];
        break;
      }
    }

    // Continue if no matching parent was found
    if (parentRefIndex == SIZE_MAX)
      continue;

    // Assign the mother-daughter linkage
    size_t daughterRefIndex = refIndexVector[i];
    reco::GenParticleRef parentRef(refProd, parentRefIndex);
    reco::GenParticleRef daughterRef(refProd, daughterRefIndex);
    genParticles[daughterRefIndex].addMother(parentRef);
    genParticles[parentRefIndex].addDaughter(daughterRef);
  }
}

size_t RHDecayTracer::getDecayParentIndex(const TrackData& parentData,
                                          std::vector<size_t>& refIndexVector,
                                          std::vector<unsigned int>& trackIdVector) {
  unsigned int parentTrackId = parentData.trackID;
  for (size_t i = 0; i < trackIdVector.size(); i++) {
    if (trackIdVector[i] == parentTrackId)
      return refIndexVector[i];
  }
  return SIZE_MAX;
}

void RHDecayTracer::addDecayDaughtersToCollection(const std::vector<TrackData>& daughtersData,
                                                  reco::GenParticleCollection& genParticles) {
  for (const auto& daughterData : daughtersData) {
    math::XYZTLorentzVector p4(daughterData.px, daughterData.py, daughterData.pz, daughterData.energy);
    math::XYZPoint vertex(daughterData.x, daughterData.y, daughterData.z);
    genParticles.emplace_back(daughterData.charge, p4, vertex, daughterData.pdgID, 1, true);
  }
}

void RHDecayTracer::updateConfiguredPDGs(const std::vector<std::string>& acceptedPDGs) {
  for (const auto& acceptedPDG : acceptedPDGs) {
    std::string s = acceptedPDG;
    // Remove whitespace
    s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
    auto dashPosition = s.find('-');
    if (dashPosition != std::string::npos) {
      int low = std::stoi(s.substr(0, dashPosition));
      int high = std::stoi(s.substr(dashPosition + 1));
      if (low > high)
        std::swap(low, high);
      pdgRanges_.emplace_back(low, high);
    } else {
      int id = std::stoi(s);
      pdgSingles_.insert(id);
    }
  }
}

bool RHDecayTracer::isConfiguredPDG(int pdgId) const {
  int absId = std::abs(pdgId);
  if (pdgSingles_.count(absId))
    return true;
  for (const auto& r : pdgRanges_) {
    if (absId >= r.first && absId <= r.second)
      return true;
  }
  return false;
}
