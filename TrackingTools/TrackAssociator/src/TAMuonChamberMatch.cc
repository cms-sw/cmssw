#include "TrackingTools/TrackAssociator/interface/TAMuonChamberMatch.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include <sstream>

int TAMuonChamberMatch::station() const {
  int muonSubdetId = id.subdetId();

  if (muonSubdetId == 1) {  //DT
    DTChamberId segId(id.rawId());
    return segId.station();
  }
  if (muonSubdetId == 2) {  //CSC
    CSCDetId segId(id.rawId());
    return segId.station();
  }
  if (muonSubdetId == 3) {  //RPC
    RPCDetId segId(id.rawId());
    return segId.station();
  }
  if (muonSubdetId == 4) {  //GEM
    GEMDetId segId(id.rawId());
    return segId.station();
  }
  if (muonSubdetId == 5) {  //ME0
    ME0DetId segId(id.rawId());
    return segId.station();
  }

  return -1;
}

std::string TAMuonChamberMatch::info() const {
  int muonSubdetId = id.subdetId();
  std::ostringstream oss;

  if (muonSubdetId == 1) {  //DT
    DTChamberId segId(id.rawId());
    oss << "DT chamber (wheel, station, sector): " << segId.wheel() << ", " << segId.station() << ", "
        << segId.sector();
  }

  if (muonSubdetId == 2) {  //CSC
    CSCDetId segId(id.rawId());
    oss << "CSC chamber (endcap, station, ring, chamber, layer): " << segId.endcap() << ", " << segId.station() << ", "
        << segId.ring() << ", " << segId.chamber() << ", " << segId.layer();
  }
  if (muonSubdetId == 3) {  //RPC
    // RPCDetId segId(id.rawId());
    oss << "RPC chamber";
  }
  if (muonSubdetId == 4) {  //GEM
    // GEMDetId segId(id.rawId());
    oss << "GEM chamber";
  }
  if (muonSubdetId == 5) {  //ME0
    // ME0DetId segId(id.rawId());
    oss << "ME0 chamber";
  }

  return oss.str();
}
