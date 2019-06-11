#include "RecoVertex/NuclearInteractionProducer/interface/NuclearLikelihood.h"

void NuclearLikelihood::calculate(const reco::Vertex& vtx) {
  likelihood_ = 0.0;
  if (vtx.isValid()) {
    if (vtx.tracksSize() > 1) {
      int idBest = 0;
      int secMaxHits = secondaryTrackMaxHits(vtx, idBest);
      if ((*(vtx.tracks_begin() + idBest))->normalizedChi2() < 3.0)
        likelihood_ = 1.0;
      else if (secMaxHits > 4)
        likelihood_ = 0.7;
      else if (secMaxHits > 3)
        likelihood_ = 0.5;
      else
        likelihood_ = 0.3;
    }
  }
}

int NuclearLikelihood::secondaryTrackMaxHits(const reco::Vertex& vtx, int& id) {
  int maxHits = 0;
  if (vtx.tracksSize() < 2)
    return 0;
  int i = 1;
  for (reco::Vertex::trackRef_iterator it = vtx.tracks_begin() + 1; it != vtx.tracks_end(); ++it) {
    int nhits = (*it)->numberOfValidHits();
    if (nhits > maxHits) {
      maxHits = nhits;
      id = i;
    }
    i++;
  }
  return maxHits;
}
