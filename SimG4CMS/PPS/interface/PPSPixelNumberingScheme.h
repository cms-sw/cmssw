#ifndef _PPS_TrackerNumberingScheme_h
#define _PPS_TrackerNumberingScheme_h 1
// -*- C++ -*-
//
// Package:     PPS
// Class  :     PPSPixelNumberingScheme
//
/**\class PPSPixelNumberingScheme PPSPixelNumberingScheme.h SimG4CMS/PPS/interface/PPSPixelNumberingScheme.h
 
 Description: This class manages the UnitID that labels PPS sensitive
              volumes
 
 Usage:
    Used in PPSPixelSD to get unique ID of sensitive detector element
 
*/
//
// Original Author:
//         Created:  Tue May 16 10:14:34 CEST 2006
//

// system include files

// user include files

#include "SimG4CMS/PPS/interface/PPSPixelOrganization.h"

class PPSPixelNumberingScheme : public PPSPixelOrganization {
public:
  // ---------- Constructor and destructor -----------------
  PPSPixelNumberingScheme();
  ~PPSPixelNumberingScheme() override;
};

#endif
