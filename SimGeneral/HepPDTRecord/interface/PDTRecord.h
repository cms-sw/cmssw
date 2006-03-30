#ifndef src_PDTRecord_h
#define src_PDTRecord_h
/**\class PDTRecord
 *
 * Description: Record for Particle Data Table entry
 *
 * \author: Luca Lista, INFN
 *
 * \version $Revision$
 *
 * $Id: PDTRecord.h,v 1.1 2006/03/14 16:22:30 llista Exp $
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class PDTRecord : 
  public edm::eventsetup::EventSetupRecordImplementation<PDTRecord> {
};

#endif
