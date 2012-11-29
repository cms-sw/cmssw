#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include <boost/thread/mutex.hpp>
#include <sstream>
#include <unistd.h>

boost::mutex                 s_mutex;
StorageAccount::StorageStats s_stats;

static double timeRealNanoSecs (void) {
#if _POSIX_TIMERS > 0
  struct timespec tm;
  if (clock_gettime(CLOCK_REALTIME, &tm) == 0)
    return tm.tv_sec * 1e9 + tm.tv_nsec;
#else
  struct timeval tm;
  if (gettimeofday(&tm, 0) == 0)
    return tm.tv_sec * 1e9 + tm.tv_usec * 1e3;
#endif
  return 0;
}

static std::string i2str(int i) {
  std::ostringstream t;
  t << i;
  return t.str();
}

static std::string d2str(double d) {
  std::ostringstream t;
  t << d;
  return t.str();
}

std::string
StorageAccount::summaryText (bool banner /*=false*/) {
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

std::string
StorageAccount::summaryXML (void) {
  std::ostringstream os;
  os << "<storage-timing-summary>\n";
  for (StorageStats::iterator i = s_stats.begin (); i != s_stats.end(); ++i)
    for (OperationStats::iterator j = i->second->begin (); j != i->second->end (); ++j)
      os << " <counter-value subsystem='" << i->first
         << "' counter-name='" << j->first
         << "' num-operations='" << j->second.attempts
         << "' num-successful-operations='" << j->second.successes
         << "' total-megabytes='" << (j->second.amount / 1024 / 1024)
         << "' total-msecs='" << (j->second.timeTotal / 1000 / 1000)
         << "' min-msecs='" << (j->second.timeMin / 1000 / 1000)
         << "' max-msecs='" << (j->second.timeMax / 1000 / 1000) << "'/>\n";
  os << "</storage-timing-summary>";
  return os.str ();
}

void
StorageAccount::fillSummary(std::map<std::string, std::string>& summary) {
  int const oneM = 1000 * 1000;
  int const oneMeg = 1024 * 1024;
  for (StorageStats::iterator i = s_stats.begin (); i != s_stats.end(); ++i) {
    for (OperationStats::iterator j = i->second->begin(); j != i->second->end(); ++j) {
      std::ostringstream os;
      os << "Timing-" << i->first << "-" << j->first << "-";
      summary.insert(std::make_pair(os.str() + "numOperations", i2str(j->second.attempts)));
      summary.insert(std::make_pair(os.str() + "numSuccessfulOperations", i2str(j->second.successes)));
      summary.insert(std::make_pair(os.str() + "totalMegabytes", d2str(j->second.amount / oneMeg)));
      summary.insert(std::make_pair(os.str() + "totalMsecs", d2str(j->second.timeTotal / oneM)));
      summary.insert(std::make_pair(os.str() + "minMsecs", d2str(j->second.timeMin / oneM)));
      summary.insert(std::make_pair(os.str() + "maxMsecs", d2str(j->second.timeMax / oneM)));
    }
  }
}

const StorageAccount::StorageStats&
StorageAccount::summary (void)
{ return s_stats; }

StorageAccount::Counter&
StorageAccount::counter (const std::string &storageClass, const std::string &operation) {
  boost::mutex::scoped_lock lock (s_mutex);
  boost::shared_ptr<OperationStats> &opstats = s_stats [storageClass];
  if (!opstats) opstats.reset(new OperationStats);

  OperationStats::iterator pos = opstats->find (operation);
  if (pos == opstats->end ()) {
    Counter x = { 0, 0, 0, 0, 0, 0, 0 };
    pos = opstats->insert (OperationStats::value_type (operation, x)).first;
  }

  return pos->second;
}

StorageAccount::Stamp::Stamp (Counter &counter)
  : m_counter (counter),
    m_start (timeRealNanoSecs ())
{
  boost::mutex::scoped_lock lock (s_mutex);
  m_counter.attempts++;
}

void
StorageAccount::Stamp::tick (double amount, int64_t count) const
{
  boost::mutex::scoped_lock lock (s_mutex);
  double elapsed = timeRealNanoSecs () - m_start;
  m_counter.successes++;

  m_counter.vector_count += count;
  m_counter.vector_square += count*count;
  m_counter.amount += amount;
  m_counter.amount_square += amount*amount;

  m_counter.timeTotal += elapsed;
  if (elapsed < m_counter.timeMin || m_counter.successes == 1)
    m_counter.timeMin = elapsed;
  if (elapsed > m_counter.timeMax)
    m_counter.timeMax = elapsed;
}
