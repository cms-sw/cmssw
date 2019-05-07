#ifndef DigiSimLinks_DTDigiSimLinkCollection_h
#define DigiSimLinks_DTDigiSimLinkCollection_h

#include <DataFormats/MuonData/interface/MuonDigiCollection.h>
#include <DataFormats/MuonDetId/interface/DTLayerId.h>
#include <SimDataFormats/DigiSimLinks/interface/DTDigiSimLink.h>

typedef MuonDigiCollection<DTLayerId, DTDigiSimLink> DTDigiSimLinkCollection;

#endif
