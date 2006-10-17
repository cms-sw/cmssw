#ifndef RFIO_ADAPTOR_RFIO_ERROR_H
# define RFIO_ADAPTOR_RFIO_ERROR_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "SealBase/IOError.h"
#include<string>

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

/** RFIO #Error object.  */
class RFIOError : public seal::IOError
{
public:
    RFIOError (const char *context, int code = 0, int scode = 0);

    virtual std::string	explainSelf (void) const;
    virtual seal::Error *clone (void) const;
    virtual void	rethrow (void);

private:
    int			m_code;
    int			m_scode;
    std::string         m_txt;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // RFIO_ADAPTOR_RFIO_ERROR_H
