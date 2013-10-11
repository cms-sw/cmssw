#include "Validation/MuonGEMDigis/interface/GEMBaseValidation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"

GEMBaseValidation::GEMBaseValidation(DQMStore* dbe,
                                               const edm::InputTag & inputTag)
{
  dbe_ = dbe;
  theInputTag = inputTag;

}


GEMBaseValidation::~GEMBaseValidation() {
 

}




