void removeCPFromPU(const std::vector<CaloParticle>& caloParticles, std::vector<size_t>& cPIndices) {
  //Consider CaloParticles coming from the hard scatterer
  //excluding the PU contribution and save the indices.
  for (unsigned int cpId = 0; cpId < caloParticles.size(); ++cpId) {
    if (caloParticles[cpId].g4Tracks()[0].eventId().event() != 0 or
        caloParticles[cpId].g4Tracks()[0].eventId().bunchCrossing() != 0) {
      LogDebug("HGCalValidator") << "Excluding CaloParticles from event: "
                                 << caloParticles[cpId].g4Tracks()[0].eventId().event()
                                 << " with BX: " << caloParticles[cpId].g4Tracks()[0].eventId().bunchCrossing()
                                 << std::endl;
      continue;
    }
    cPIndices.emplace_back(cpId);
  }
}
