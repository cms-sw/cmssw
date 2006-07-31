#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "SealBase/LongLong.h"
# include <boost/shared_ptr.hpp>
# include <string>
# include <map>
# include <boost/thread/recursive_mutex.hpp>
# include <boost/thread/mutex.hpp>



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
    inline Counter() : attempts(0),successes(0), amount(0), 
	time_tot(0), time_min(10E10), time_max(0){}
    seal::ULongLong	attempts;
    seal::ULongLong	successes;
    double		amount;
    double              time_tot;
    double              time_min;
    double		time_max;
    std::string         idTag;
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
  friend class Stamp;

  typedef std::map<std::string, Counter> OperationStats;
  typedef std::map<std::string, boost::shared_ptr<OperationStats> > StorageStats;
  
  static const StorageStats &	summary (void);
  static std::string		summaryText (void);
  static Counter &		counter (const std::string &storageClass,
					 const std::string &operation);
  
  struct LastOp
  {
    std::string     idTag;
    double	    startTime;
    double	    elapsed;
  };
  
  
  static  LastOp & lastOp();
  static  void setCurrentOp(const Counter * currOp, double stime);
  
private:
  static boost::mutex s_mutex;
  static StorageStats	s_stats;

};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_H
