#ifndef src_PDTRecord_h
#define src_PDTRecord_h
/**\class PDTRecord
 *
 * Description: Record for Particle Data Table entry
 *
 * Author: Luca Lista, INFN
 * Created:     Fri Mar 10 16:31:53 CET 2006
 * $Id: PDTRecord.h,v 1.1 2006/03/13 18:03:05 llista Exp $
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class PDTRecord : 
  public edm::eventsetup::EventSetupRecordImplementation<PDTRecord> {
};

#endif
