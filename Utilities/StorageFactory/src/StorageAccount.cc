//<<<<<< INCLUDES                                                       >>>>>>

#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "SealBase/TimeInfo.h"
#include <sstream>
#include <boost/thread/recursive_mutex.hpp>
#include <boost/thread/mutex.hpp>

//<<<<<< PRIVATE DEFINES                                                >>>>>>
//<<<<<< PRIVATE CONSTANTS                                              >>>>>>
//<<<<<< PRIVATE TYPES                                                  >>>>>>
//<<<<<< PRIVATE VARIABLE DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC VARIABLE DEFINITIONS                                    >>>>>>
//<<<<<< CLASS STRUCTURE INITIALIZATION                                 >>>>>>

boost::mutex			s_mutex;
StorageAccount::StorageStats	s_stats;

//<<<<<< PRIVATE FUNCTION DEFINITIONS                                   >>>>>>
//<<<<<< PUBLIC FUNCTION DEFINITIONS                                    >>>>>>
//<<<<<< MEMBER FUNCTION DEFINITIONS                                    >>>>>>

std::string
StorageAccount::summaryText (bool banner /*=false*/)
{
  bool first = true;
  std::ostringstream os;
  if (banner) 
    os << "stats: class/operation/attempts/successes/amount/time-total/time-min/time-max\n";
  for (StorageStats::iterator i = s_stats.begin (); i != s_stats.end(); ++i)
    for (OperationStats::iterator j = i->second->begin (); j != i->second->end (); ++j, first = false)
      os << (first ? "" : "; ")
	 << i->first << '/'
	 << j->first << '='
	 << j->second.attempts << '/'
	 << j->second.successes << '/'
	 << (j->second.amount / 1024 / 1024) << "MB/"
	 << (j->second.timeTotal / 1000 / 1000) << "ms/" 
         << (j->second.timeMin / 1000 / 1000) << "ms/" 
         << (j->second.timeMax / 1000 / 1000) << "ms";
  
  return os.str ();
}

const StorageAccount::StorageStats &
StorageAccount::summary (void)
{ return s_stats; }

StorageAccount::Counter &
StorageAccount::counter (const std::string &storageClass, const std::string &operation)
{
  boost::mutex::scoped_lock	    lock (s_mutex);
  boost::shared_ptr<OperationStats> &opstats = s_stats [storageClass];
  if (! opstats) opstats.reset(new OperationStats);
  
  OperationStats::iterator pos = opstats->find (operation);
  if (pos == opstats->end ())
    {
      Counter x = { 0, 0, 0, 0, 0 };
      pos = opstats->insert (OperationStats::value_type (operation, x)).first;
    }
  
  return pos->second;
}

StorageAccount::Stamp::Stamp (Counter &counter)
  : m_counter (counter),
    m_start (seal::TimeInfo::realNsecs ())
{
  boost::mutex::scoped_lock lock (s_mutex);
  m_counter.attempts++;
}

void
StorageAccount::Stamp::tick (double amount) const
{
  boost::mutex::scoped_lock lock (s_mutex);
  double elapsed = seal::TimeInfo::realNsecs () - m_start;
  m_counter.successes++;
  m_counter.amount += amount;
  m_counter.timeTotal += elapsed;
  if (elapsed < m_counter.timeMin) m_counter.timeMin = elapsed;
  if (elapsed > m_counter.timeMax) m_counter.timeMax = elapsed;
}
