#include "TrackingTools/TrackAssociator/interface/MuonChamberMatch.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include <sstream>

int MuonChamberMatch::station() const {
	int muonSubdetId = id.subdetId();

	if(muonSubdetId==1) {//DT
		DTChamberId segId(id.rawId());
		return segId.station();
	}
	if(muonSubdetId==2) {//CSC
		CSCDetId segId(id.rawId());
		return segId.station();
	}
	if(muonSubdetId==3) {//RPC
		RPCDetId segId(id.rawId());
		return segId.station();
	}

	return -1;
}

std::string MuonChamberMatch::info() const {
   int muonSubdetId = id.subdetId();
   std::ostringstream oss;

   if(muonSubdetId==1) {//DT
      DTChamberId segId(id.rawId());
      oss << "DT chamber (wheel, station, sector): "
	<< segId.wheel() << ", "
	<< segId.station() << ", "
	<< segId.sector();
   }
	
   if(muonSubdetId==2) {//CSC
      CSCDetId segId(id.rawId());
      oss << "CSC chamber (endcap, station, ring, chamber, layer): "
	<< segId.endcap() << ", "
	<< segId.station() << ", "
	<< segId.ring() << ", "
	<< segId.chamber() << ", "
	<< segId.layer();
   }
   if(muonSubdetId==3) {//RPC
      // RPCDetId segId(id.rawId());
      oss << "RPC chamber";
   }

   return oss.str();
}
