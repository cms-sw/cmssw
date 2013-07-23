#ifndef src_PDTRecord_h
#define src_PDTRecord_h
/**\class PDTRecord
 *
 * Description: Record for Particle Data Table entry
 *
 * \author: Luca Lista, INFN
 *
 * \version $Revision: 1.2 $
 *
 * $Id: PDTRecord.h,v 1.2 2006/03/30 12:21:24 llista Exp $
 */

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class PDTRecord : 
  public edm::eventsetup::EventSetupRecordImplementation<PDTRecord> {
};

#endif
