#ifndef DCACHE_ADAPTOR_DCACHE_ERROR_H
# define DCACHE_ADAPTOR_DCACHE_ERROR_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "SealBase/IOError.h"

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

/** DCache #Error object.  */
class DCacheError : public seal::IOError
{
public:
    DCacheError (const char *context, int code = 0);

    virtual std::string	explainSelf (void) const;
    virtual seal::Error *clone (void) const;
    virtual void	rethrow (void);

private:
    int			m_code;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // DCACHE_ADAPTOR_DCACHE_ERROR_H
