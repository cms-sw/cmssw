//
// Package:         TrackingTools/RoadSearchDetIdHelper
// Class:           RoadSearchDetIdHelper
// 
// Description:     helper functions concerning DetIds
//
// Original Author: Oliver Gutsche, gutsche@fnal.gov
// Created:         Sun Jan 28 19:06:20 UTC 2007
//
// $Author: gutsche $
// $Date: 2006/11/10 21:54:49 $
// $Revision: 1.20 $
//

#include "TrackingTools/RoadSearchHitAccess/interface/RoadSearchDetIdHelper.h"

#include <sstream>

RoadSearchDetIdHelper::RoadSearchDetIdHelper() {
}

RoadSearchDetIdHelper::~RoadSearchDetIdHelper() {
}

std::string RoadSearchDetIdHelper::Print(const DetId id) {
  //
  // print DetId composition according to the type
  //

  std::ostringstream output;

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    TIBDetId tibid(id.rawId()); 
    output << "TIB ring Detid: " << id.rawId() 
	   << " layer: " << tibid.layer() 
	   << " fw(0)/bw(1): " << tibid.string()[0]
	   << " int(0)/ext(1): " << tibid.string()[1] 
	   << " string: " << tibid.string()[2] 
	   << " module: " << tibid.module();
    if ( IsMatched(tibid) ) {
      output << " corresponding to matched detunit of glued sensor";
    } else if ( IsGluedRPhi(tibid) ) {
      output << " corresponding to rphi detunit of glued sensor";
    } else if ( IsStereo(tibid) ) {
      output << " corresponding to stereo detunit of glued sensor";
    } else if ( IsSingleRPhi(tibid) ) {
      output << " corresponding to rphi detunit of single sensor";
    }
    output << std::endl; 
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    TOBDetId tobid(id.rawId()); 
    output << "TOB ring Detid: " << id.rawId() 
	   << " layer: " << tobid.layer() 
	   << " fw(0)/bw(1): " << tobid.rod()[0]
	   << " rod: " << tobid.rod()[1] 
	   << " detector: " << tobid.module();
    if ( IsMatched(tobid) ) {
      output << " corresponding to matched detunit of glued sensor";
    } else if ( IsGluedRPhi(tobid) ) {
      output << " corresponding to rphi detunit of glued sensor";
    } else if ( IsStereo(tobid) ) {
      output << " corresponding to stereo detunit of glued sensor";
    } else if ( IsSingleRPhi(tobid) ) {
      output << " corresponding to rphi detunit of single sensor";
    }
    output << std::endl; 
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    TIDDetId tidid(id.rawId()); 
    output << "TID ring Detid: " << id.rawId() 
	   << " side neg(1)/pos(2): " << tidid.side() 
	   << " wheel: " << tidid.wheel()
	   << " ring: " << tidid.ring()
	   << " detector fw(0)/bw(1): " << tidid.module()[0]
	   << " detector: " << tidid.module()[1];
    if ( IsMatched(tidid) ) {
      output << " corresponding to matched detunit of glued sensor";
    } else if ( IsGluedRPhi(tidid) ) {
      output << " corresponding to rphi detunit of glued sensor";
    } else if ( IsStereo(tidid) ) {
      output << " corresponding to stereo detunit of glued sensor";
    } else if ( IsSingleRPhi(tidid) ) {
      output << " corresponding to rphi detunit of single sensor";
    }
    output << std::endl; 
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    TECDetId tecid(id.rawId()); 
    output << "TEC ring DetId: " << id.rawId() 
	   << " side neg(1)/pos(2): " << tecid.side() 
	   << " wheel: " << tecid.wheel()
	   << " petal fw(0)/bw(1): " << tecid.petal()[0]
	   << " petal: " << tecid.petal()[1] 
	   << " ring: " << tecid.ring()
	   << " module: " << tecid.module();
    if ( IsMatched(tecid) ) {
      output << " corresponding to matched detunit of glued sensor";
    } else if ( IsGluedRPhi(tecid) ) {
      output << " corresponding to rphi detunit of glued sensor";
    } else if ( IsStereo(tecid) ) {
      output << " corresponding to stereo detunit of glued sensor";
    } else if ( IsSingleRPhi(tecid) ) {
      output << " corresponding to rphi detunit of single sensor";
    }
    output << std::endl; 
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
    PXBDetId pxbid(id.rawId()); 
    output << "PXB ring DetId: " << id.rawId() 
	   << " layer: " << pxbid.layer()
	   << " ladder: " << pxbid.ladder()
	   << " detector: " << pxbid.module()
	   << std::endl; 
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap ) {
    PXFDetId pxfid(id.rawId()); 
    output << "PXF ring DetId: " << id.rawId() 
	   << " side: " << pxfid.side()
	   << " disk: " << pxfid.disk()
	   << " blade: " << pxfid.blade()
	   << " detector: " << pxfid.module()
	   << std::endl; 
  }


  return output.str();
}

bool RoadSearchDetIdHelper::IsMatched(const DetId id) {

  // return value
  bool result = false;

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    TIBDetId tibid(id.rawId()); 
    if ( !tibid.glued() ) {
      if ( tibid.layer() == 1 ||
	   tibid.layer() == 2 ) {
	result = true;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    TOBDetId tobid(id.rawId()); 
    if ( !tobid.glued() ) {
      if ( tobid.layer() == 1 ||
	   tobid.layer() == 2 ) {
	result = true;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    TIDDetId tidid(id.rawId()); 
    if ( !tidid.glued() ) {
      if ( tidid.ring() == 1 ||
	   tidid.ring() == 2 ) {
	result = true;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    TECDetId tecid(id.rawId()); 
    if ( !tecid.glued() ) {
      if ( tecid.ring() == 1 ||
	   tecid.ring() == 2 ||
	   tecid.ring() == 5 ) {
	result = true;
      }
    }
  }

  return result;
}

bool RoadSearchDetIdHelper::IsSingleRPhi(const DetId id) {

  // return value
  bool result = true;

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    TIBDetId tibid(id.rawId()); 
    if ( !tibid.glued() ) {
      if ( tibid.layer() == 1 ||
	   tibid.layer() == 2 ) {
	result = false;
      }
    } else {
      result = false;
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    TOBDetId tobid(id.rawId()); 
    if ( !tobid.glued() ) {
      if ( tobid.layer() == 1 ||
	   tobid.layer() == 2 ) {
	result = false;
      }
    } else {
      if ( (tobid.rawId()-2) != tobid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    TIDDetId tidid(id.rawId()); 
    if ( !tidid.glued() ) {
      if ( tidid.ring() == 1 ||
	   tidid.ring() == 2 ) {
	result = false;
      } 
    } else {
      if ( (tidid.rawId()-2) != tidid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    TECDetId tecid(id.rawId()); 
    if ( !tecid.glued() ) {
      if ( tecid.ring() == 1 ||
	   tecid.ring() == 2 ||
	   tecid.ring() == 5 ) {
	result = false;
      } 
    } else {
      if ( (tecid.rawId()-2) != tecid.glued() ) {
	result = false;
      }
    }
  }

  return result;
}

bool RoadSearchDetIdHelper::IsGluedRPhi(const DetId id) {

  // return value
  bool result = true;

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    TIBDetId tibid(id.rawId()); 
    if ( !tibid.glued() ) {
      result = false;
    } else {
      if ( (tibid.rawId()-2) != tibid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    TOBDetId tobid(id.rawId()); 
    if ( !tobid.glued() ) {
      result = false;
    } else {
      if ( (tobid.rawId()-2) != tobid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    TIDDetId tidid(id.rawId()); 
    if ( !tidid.glued() ) {
      result = false;
    } else {
      if ( (tidid.rawId()-2) != tidid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    TECDetId tecid(id.rawId()); 
    if ( !tecid.glued() ) {
      result = false;
    } else {
      if ( (tecid.rawId()-2) != tecid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
    result = false;
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap ) {
    result = false;
  } 

  return result;
}

bool RoadSearchDetIdHelper::IsStereo(const DetId id) {

  // return value
  bool result = true;

  if ( (unsigned int)id.subdetId() == StripSubdetector::TIB ) {
    TIBDetId tibid(id.rawId()); 
    if ( !tibid.glued() ) {
      result = false;
    } else {
      if ( (tibid.rawId()-1) != tibid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TOB ) {
    TOBDetId tobid(id.rawId()); 
    if ( !tobid.glued() ) {
      result = false;
    } else {
      if ( (tobid.rawId()-1) != tobid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TID ) {
    TIDDetId tidid(id.rawId()); 
    if ( !tidid.glued() ) {
      result = false;
    } else {
      if ( (tidid.rawId()-1) != tidid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == StripSubdetector::TEC ) {
    TECDetId tecid(id.rawId()); 
    if ( !tecid.glued() ) {
      result = false;
    } else {
      if ( (tecid.rawId()-1) != tecid.glued() ) {
	result = false;
      }
    }
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelBarrel ) {
    result = false;
  } else if ( (unsigned int)id.subdetId() == PixelSubdetector::PixelEndcap ) {
    result = false;
  }

  return result;
}

DetId RoadSearchDetIdHelper::ReturnRPhiId(const DetId id) {
  //
  // return corresponding rphi id
  //

  if ( IsMatched(id) ) {
    return DetId(id.rawId()+2);
  } else if ( IsStereo(id) ) {
    return DetId(id.rawId()+1);
  } else {
    return id;
  }

  return id;
}

