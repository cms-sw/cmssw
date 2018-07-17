#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#ifdef EDM_ML_DEBUG

#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"

namespace {
   inline
   void dump(TrackingRecHit const & hit, int hitcounter, const std::string& msgCat) {
    if (hit.isValid()) {
      LogTrace(msgCat)<< " ----------------- HIT #" << hitcounter << " (VALID)-----------------------\n"
	<< "  HIT IS AT R   " << hit.globalPosition().perp() << "\n"
	<< "  HIT IS AT Z   " << hit.globalPosition().z() << "\n"
	<< "  HIT IS AT Phi " << hit.globalPosition().phi() << "\n"
	<< "  HIT IS AT Loc " << hit.localPosition() << "\n"
	<< "  WITH LocError " << hit.localPositionError() << "\n"
	<< "  HIT IS AT Glo " << hit.globalPosition() << "\n"
	<< "SURFACE POSITION" << "\n"
	<< hit.surface()->position()<<"\n"
	<< "SURFACE ROTATION" << "\n"
	<< hit.surface()->rotation()
        <<  "dimension " << hit.dimension();

      DetId hitId = hit.geographicalId();

      LogDebug(msgCat) << " hit det=" << hitId.rawId();

      if(hitId.det() == DetId::Tracker) {
	switch(hitId.subdetId()) {
	  case StripSubdetector::TIB:
	    LogDebug(msgCat) << " I am TIB"; break;
	  case StripSubdetector::TOB:
	    LogDebug(msgCat) << " I am TOB"; break;
	  case StripSubdetector::TEC:
	    LogDebug(msgCat) << " I am TEC"; break;
	  case StripSubdetector::TID:
	    LogDebug(msgCat) << " I am TID"; break;
	  case PixelSubdetector::PixelBarrel:
	    LogDebug(msgCat) << " I am PixBar"; break;
	  case PixelSubdetector::PixelEndcap:
	    LogDebug(msgCat) << " I am PixFwd"; break;
	  default:
	    LogDebug(msgCat) << " UNKNOWN TRACKER HIT TYPE ";
	}
      }
      else if(hitId.det() == DetId::Muon) {
	if(hitId.subdetId() == MuonSubdetId::DT)
	  LogDebug(msgCat) << " I am DT " << DTWireId(hitId);
	else if (hitId.subdetId() == MuonSubdetId::CSC )
	  LogDebug(msgCat) << " I am CSC " << CSCDetId(hitId);
	else if (hitId.subdetId() == MuonSubdetId::RPC )
	  LogDebug(msgCat) << " I am RPC " << RPCDetId(hitId);
	else if (hitId.subdetId() == MuonSubdetId::GEM )
	  LogDebug(msgCat) << " I am GEM " << GEMDetId(hitId);

	else if (hitId.subdetId() == MuonSubdetId::ME0 )
	  LogDebug(msgCat) << " I am ME0 " << ME0DetId(hitId);
	else
	  LogDebug(msgCat) << " UNKNOWN MUON HIT TYPE ";
      }
      else
	LogDebug(msgCat) << " UNKNOWN HIT TYPE ";

    } else {
      LogDebug(msgCat)
	<< " ----------------- INVALID HIT #" << hitcounter << " -----------------------";
    }
   }
#include <sstream>
   inline void dump(TrajectoryStateOnSurface const & tsos, const char * header, const std::string& msgCat) {
     std::ostringstream ss; ss<< " weights ";
     for (auto const & c : tsos.components()) ss << c.weight() << '/';
     ss << "\nmomentums ";
     for (auto const & c : tsos.components()) ss << c.globalMomentum().mag() << '/';
     ss << "\ndeltap/p ";
    for (auto const & c : tsos.components()) ss << std::sqrt(tsos.curvilinearError().matrix()(0,0))/c.globalMomentum().mag() << '/';
     LogTrace(msgCat)
      << header  << "! size " << tsos.components().size() << ss.str() << "\n"
      <<" with local position " << tsos.localPosition() << "\n"
      << tsos;
   }
}
#else
namespace {
   inline void dump(TrackingRecHit const &, int, const std::string&) {}
   inline void dump(TrajectoryStateOnSurface const &, const char *, const std::string&){}
}
#endif
