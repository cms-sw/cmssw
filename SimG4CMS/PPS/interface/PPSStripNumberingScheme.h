#ifndef PPSStripNumberingScheme_h
#define PPSStripNumberingScheme_h

#include "SimG4CMS/PPS/interface/PPSStripOrganization.h"

class PPSStripNumberingScheme : public PPSStripOrganization {
public:
  PPSStripNumberingScheme(int i);
  ~PPSStripNumberingScheme() override;
};

#endif  //PPS_PPSStripNumberingScheme_h
