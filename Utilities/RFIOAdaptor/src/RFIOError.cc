//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/RFIOAdaptor/interface/RFIOError.h"
#include "SealBase/StringFormat.h"

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>
//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

RFIOError::RFIOError (const char *context, int code /* = 0 */, int scode /* = 0 */)
    : IOError (context),
      m_code (code),
      m_scode (scode)
{}

std::string
RFIOError::explainSelf (void) const
{ return seal::StringFormat ("RFIO error %1/%2").arg (m_code).arg (m_scode); }

seal::Error *
RFIOError::clone (void) const
{ return new RFIOError (*this); }

void
RFIOError::rethrow (void)
{ throw *this; }
