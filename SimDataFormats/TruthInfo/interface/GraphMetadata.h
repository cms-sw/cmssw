// Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>
//
// How the truth graph of a run was built: one product per run, so a reader can tell what
// a graph holds without the configuration of the job that wrote it.

#ifndef SimDataFormats_TruthInfo_GraphMetadata_h
#define SimDataFormats_TruthInfo_GraphMetadata_h

#include <cstdint>
#include <string>
#include <vector>

namespace truth {

  struct GraphMetadata {
    enum class BuildMode : uint8_t {
      Unknown = 0,
      // The GEN record only: no SIM node, no hit.
      GenOnly = 1,
      // GEN and SIM of one interaction per event, without pile-up.
      GenSim = 2,
      // GEN and SIM of the signal and the pile-up, built during mixing.
      GenSimMixed = 3,
    };

    BuildMode buildMode = BuildMode::Unknown;
    // The bunch crossings whose pile-up is in the graph. GenSimMixed only.
    std::vector<int32_t> pileupBunchCrossings;
    // The release of the job that built the logical graph. Empty when that job is not found.
    std::string cmsswVersion;

    // The generator: module type, program name and version, generator fragment, release
    // of the job that ran it, and the configuration of the generator module.
    // An empty field is not known to the job that wrote this product.
    std::string generatorModule;
    std::string generatorName;
    std::string generatorVersion;
    std::string generatorFragment;
    std::string generatorCmsswVersion;
    std::string generatorConfig;

    // The selection and the pruning of the logical graph. A parameter that the job which built
    // the graph did not have keeps its default value here.
    std::string presetName;
    std::vector<int32_t> signalSeedPdgIds;
    std::vector<int32_t> seedHadronFlavors;
    std::vector<int32_t> reconstructablePdgIds;
    bool collapseIntermediateGenParticles = false;
    bool dropHitlessSimSubgraphs = false;
    // The sim-hit collections that decide which SIM particle has a hit.
    std::vector<std::string> pruningSimHitCollections;
    // The configuration of the logical and the raw graph producers.
    std::string graphProducerConfig;
    std::string rawGraphProducerConfig;

    // The name of each bit of ParticleData::levelFlags, in step.
    std::vector<std::string> levelNames;
    std::vector<uint32_t> levelFlags;

    // The hit index: its channels, its inputs (sim-hit collections and digi-sim links), and
    // the energy unit of each subdetector, in step. Empty when the job builds no hit index.
    std::vector<std::string> hitIndexChannels;
    std::vector<std::string> hitIndexInputs;
    std::vector<std::string> energySubdetectors;
    std::vector<std::string> energyConventions;
    std::string hitIndexConfig;

    bool operator==(GraphMetadata const&) const = default;
    // Two files of the same run must carry the same metadata; the framework keeps the
    // first and logs an error.
    [[nodiscard]] bool isProductEqual(GraphMetadata const& other) const { return *this == other; }
  };

}  // namespace truth

#endif
