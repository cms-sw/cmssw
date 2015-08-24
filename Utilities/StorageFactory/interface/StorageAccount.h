#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_H

# include <boost/shared_ptr.hpp>
# include <stdint.h>
# include <string>
# include <chrono>
# include <atomic>
# include <map>
# include "tbb/concurrent_unordered_map.h"

class StorageAccount {
public:
  
  enum class Operation {
    check,
    close,
    construct,
    destruct,
    flush,
    open,
    position,
    prefetch,
    read,
    readActual,
    readAsync,
    readPrefetchToCache,
    readViaCache,
    readv,
    resize,
    seek,
    stagein,
    stat,
    write,
    writeActual,
    writeViaCache,
    writev
  };
  
  struct Counter {
    Counter():
    attempts{0},
    successes{0},
    amount{0},
    amount_square{0.},
    vector_count{0},
    vector_square{0},
    timeTotal{0.},
    timeMin{0.},
    timeMax{0.} {}

    Counter(Counter&& ) = default;

    //NOTE: This is needed by tbb::concurrent_unordered_map when it
    // is constructing a new one. This would not give correct results
    // if the object being passed were being updated, but that is not
    // the case for operator[]
    Counter( Counter const& iOther):
    attempts{iOther.attempts.load()},
    successes{iOther.successes.load()},
    amount{iOther.amount.load()},
    amount_square{iOther.amount_square.load()},
    vector_count{iOther.vector_count.load()},
    vector_square{iOther.vector_square.load()},
    timeTotal{iOther.timeTotal.load()},
    timeMin{iOther.timeMin.load()},
    timeMax{iOther.timeMax.load()} {}
    
    //Use atomics to allow concurrent read/write for intermediate
    // output of the statics while running. The values obtained
    // won't be completely consistent but should be good enough for
    // monitoring. The values obtained once the program is shutting
    // down should be completely consistent.
    
    std::atomic<uint64_t> attempts;
    std::atomic<uint64_t> successes;
    std::atomic<uint64_t> amount;
    // NOTE: Significant risk exists for underflow in this value.
    // However, the use cases are marginal so I let it pass.
    std::atomic<double>   amount_square;
    std::atomic<int64_t>  vector_count;
    std::atomic<int64_t>  vector_square;
    std::atomic<double>   timeTotal;
    std::atomic<double>   timeMin;
    std::atomic<double>   timeMax;
    
    static void addTo(std::atomic<double>& iAtomic, double iToAdd) {
      double oldValue = iAtomic.load();
      double newValue = oldValue + iToAdd;
      while( not iAtomic.compare_exchange_weak(oldValue, newValue)) {
        newValue = oldValue+iToAdd;
      }
    }
  };

  class Stamp {
  public:
    Stamp (Counter &counter);

    void     tick (uint64_t amount = 0, int64_t tick = 0) const;
  protected:
    Counter &m_counter;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
  };

  class StorageClassToken {
  public:
    StorageClassToken(StorageClassToken const&) = default;
    int value() const { return m_value;}

    friend class StorageAccount;
  private:
    StorageClassToken() = delete;
    explicit StorageClassToken(int iValue) : m_value{iValue} {}

    int m_value;
    
  };
  
  typedef tbb::concurrent_unordered_map<int, Counter> OperationStats;
  typedef tbb::concurrent_unordered_map<int, OperationStats > StorageStats;

  static char const* operationName(Operation operation);
  static StorageClassToken tokenForStorageClassName( std::string const& iName);
  static const std::string& nameForToken( StorageClassToken);
  
  static const StorageStats& summary(void);
  static std::string         summaryText(bool banner=false);
  static void                fillSummary(std::map<std::string, std::string> &summary);
  static Counter&            counter (StorageClassToken token,
                                      Operation operation);

private:
  static StorageStats m_stats;

};

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_H
