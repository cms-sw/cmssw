//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/plugins/RFIOShiftStorageMaker.h"
#include "Utilities/StorageFactory/interface/StorageMakerFactory.h"

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

using edm::storage::StorageMakerFactory;
DEFINE_EDM_PLUGIN (StorageMakerFactory, RFIOShiftStorageMaker, "rfio");
