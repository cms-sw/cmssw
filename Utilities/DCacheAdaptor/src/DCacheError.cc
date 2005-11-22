//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/DCacheAdaptor/interface/DCacheError.h"
#include "SealBase/StringFormat.h"
#include <dcap.h>

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

DCacheError::DCacheError (const char *context, int code /* = 0 */)
    : IOError (context),
      m_code (code)
{}

std::string
DCacheError::explainSelf (void) const
{ return seal::StringFormat ("DCache error %1: %2")
	 .arg (m_code).arg (dc_strerror (m_code)); }

seal::Error *
DCacheError::clone (void) const
{ return new DCacheError (*this); }

void
DCacheError::rethrow (void)
{ throw *this; }
