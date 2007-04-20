//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/HttpStorageMaker.h"
#include "Utilities/StorageFactory/interface/GsiFTPStorageMaker.h"
#include "Utilities/StorageFactory/interface/LocalStorageMaker.h"
#include "Utilities/StorageFactory/interface/ZipMemberStorageMaker.h"
#include "Utilities/StorageFactory/interface/DCacheStorageMaker.h"
#include "Utilities/StorageFactory/interface/RFIOStorageMaker.h"
#include "Utilities/StorageFactory/interface/RedirectStorageMaker.h"
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
DEFINE_EDM_PLUGIN (StorageMakerFactory, HttpStorageMaker, "http");
DEFINE_EDM_PLUGIN (StorageMakerFactory, HttpStorageMaker, "ftp");
DEFINE_EDM_PLUGIN (StorageMakerFactory, HttpStorageMaker, "web");
DEFINE_EDM_PLUGIN (StorageMakerFactory, GsiFTPStorageMaker, "gsiftp");
DEFINE_EDM_PLUGIN (StorageMakerFactory, GsiFTPStorageMaker, "sfn");
DEFINE_EDM_PLUGIN (StorageMakerFactory, LocalStorageMaker, "file");
DEFINE_EDM_PLUGIN (StorageMakerFactory, ZipMemberStorageMaker, "zip-member");
DEFINE_EDM_PLUGIN (StorageMakerFactory, DCacheStorageMaker, "dcache");
DEFINE_EDM_PLUGIN (StorageMakerFactory, DCacheStorageMaker, "dcap");
DEFINE_EDM_PLUGIN (StorageMakerFactory, DCacheStorageMaker, "gsidcap");
DEFINE_EDM_PLUGIN (StorageMakerFactory, RFIOStorageMaker, "rfio");
DEFINE_EDM_PLUGIN (StorageMakerFactory, RedirectStorageMaker, "redirect");
