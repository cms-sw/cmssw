#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include <cassert>
#include <mutex>
#include <sstream>
#include <unistd.h>
#include <sys/time.h>
using namespace edm::storage;

namespace {
  char const* const kOperationNames[] = {
      "check",        "close",       "construct",     "destruct",   "flush",     "open",
      "position",     "prefetch",    "read",          "readActual", "readAsync", "readPrefetchToCache",
      "readViaCache", "readv",       "resize",        "seek",       "stagein",   "stat",
      "write",        "writeActual", "writeViaCache", "writev"};

  //Storage class names to the value of the token to which they are assigned
  oneapi::tbb::concurrent_unordered_map<std::string, int> s_nameToToken;
  std::atomic<int> s_nextTokenValue{0};
}  // namespace

StorageAccount::StorageStats StorageAccount::m_stats;

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

inline char const* StorageAccount::operationName(Operation operation) {
  return kOperationNames[static_cast<int>(operation)];
}

StorageAccount::StorageClassToken StorageAccount::tokenForStorageClassName(std::string const& iName) {
  auto itFound = s_nameToToken.find(iName);
  if (itFound != s_nameToToken.end()) {
    return StorageClassToken(itFound->second);
  }
  int value = s_nextTokenValue++;

  s_nameToToken.insert(std::make_pair(iName, value));

  return StorageClassToken(value);
}

const std::string& StorageAccount::nameForToken(StorageClassToken iToken) {
  for (auto it = s_nameToToken.begin(), itEnd = s_nameToToken.end(); it != itEnd; ++it) {
    if (it->second == iToken.value()) {
      return it->first;
    }
  }
  assert(false);
}

std::string StorageAccount::summaryText(bool banner /*=false*/) {
  bool first = true;
  std::ostringstream os;
  if (banner)
    os << "stats: class/operation/attempts/successes/amount/time-total/time-min/time-max\n";
  for (auto i = s_nameToToken.begin(); i != s_nameToToken.end(); ++i) {
    auto const& opStats = m_stats[i->second];
    for (auto j = opStats.begin(); j != opStats.end(); ++j, first = false)
      os << (first ? "" : "; ") << (i->first) << '/' << kOperationNames[j->first] << '=' << j->second.attempts << '/'
         << j->second.successes << '/' << (static_cast<double>(j->second.amount) / 1024 / 1024) << "MB/"
         << (static_cast<double>(j->second.timeTotal) / 1000 / 1000) << "ms/"
         << (static_cast<double>(j->second.timeMin) / 1000 / 1000) << "ms/"
         << (static_cast<double>(j->second.timeMax) / 1000 / 1000) << "ms";
  }
  return os.str();
}

void StorageAccount::fillSummary(std::map<std::string, std::string>& summary) {
  int const oneM = 1000 * 1000;
  int const oneMeg = 1024 * 1024;
  for (auto i = s_nameToToken.begin(); i != s_nameToToken.end(); ++i) {
    auto const& opStats = m_stats[i->second];
    for (auto j = opStats.begin(); j != opStats.end(); ++j) {
      std::ostringstream os;
      os << "Timing-" << i->first << "-" << kOperationNames[j->first] << "-";
      summary.insert(std::make_pair(os.str() + "numOperations", i2str(j->second.attempts)));
      summary.insert(std::make_pair(os.str() + "numSuccessfulOperations", i2str(j->second.successes)));
      summary.insert(
          std::make_pair(os.str() + "totalMegabytes", d2str(static_cast<double>(j->second.amount) / oneMeg)));
      summary.insert(std::make_pair(os.str() + "totalMsecs", d2str(static_cast<double>(j->second.timeTotal) / oneM)));
      summary.insert(std::make_pair(os.str() + "minMsecs", d2str(static_cast<double>(j->second.timeMin) / oneM)));
      summary.insert(std::make_pair(os.str() + "maxMsecs", d2str(static_cast<double>(j->second.timeMax) / oneM)));
    }
  }
}

const StorageAccount::StorageStats& StorageAccount::summary() { return m_stats; }

StorageAccount::Counter& StorageAccount::counter(StorageClassToken token, Operation operation) {
  auto& opstats = m_stats[token.value()];

  return opstats[static_cast<int>(operation)];
}

StorageAccount::Stamp::Stamp(Counter& counter) : m_counter(counter), m_start(std::chrono::steady_clock::now()) {
  m_counter.attempts++;
}

void StorageAccount::Stamp::tick(uint64_t amount, int64_t count) const {
  std::chrono::nanoseconds elapsed_ns = std::chrono::steady_clock::now() - m_start;
  uint64_t elapsed = elapsed_ns.count();
  m_counter.successes++;

  m_counter.vector_count += count;
  m_counter.vector_square += count * count;
  m_counter.amount += amount;
  Counter::addTo(m_counter.amount_square, amount * amount);

  Counter::addTo(m_counter.timeTotal, elapsed);
  if (elapsed < m_counter.timeMin || m_counter.successes == 1)
    m_counter.timeMin = elapsed;
  if (elapsed > m_counter.timeMax)
    m_counter.timeMax = elapsed;
}
