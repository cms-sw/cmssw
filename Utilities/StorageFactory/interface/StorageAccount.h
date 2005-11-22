#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "SealBase/LongLong.h"
# include <boost/shared_ptr.hpp>
# include <string>
# include <map>

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>
//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

class StorageAccount
{
public:
    struct Counter
    {
	seal::ULongLong	attempts;
	seal::ULongLong	successes;
	double		amount;
	double		time;
    };
    class Stamp
    {
    public:
        Stamp (Counter &counter);

        void		tick (double amount = 0.) const;

    private:
	Counter		&m_counter;
        double		m_start;
    };

    typedef std::map<std::string, Counter> OperationStats;
    typedef std::map<std::string, boost::shared_ptr<OperationStats> > StorageStats;

    static const StorageStats &	summary (void);
    static std::string		summaryText (void);
    static Counter &		counter (const std::string &storageClass,
					 const std::string &operation);

private:
    static StorageStats	s_stats;
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_H
