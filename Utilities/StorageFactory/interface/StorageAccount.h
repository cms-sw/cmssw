#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_H

# include <boost/shared_ptr.hpp>
# include <stdint.h>
# include <string>
# include <map>
# include <mutex>
# include <chrono>
# include <atomic>

class StorageAccount {
public:
  struct Counter {
    Counter(Counter const&) = delete;
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

  typedef std::map<std::string, Counter> OperationStats;
  typedef std::map<std::string, OperationStats > StorageStats;

  class StorageStatsSentry {
    friend class StorageAccount;

  public:
    StorageStatsSentry(StorageStatsSentry const&) = delete;
    StorageStatsSentry& operator=(StorageStatsSentry const&) = delete;
    StorageStatsSentry(StorageStatsSentry &&) = default;

    StorageStats* operator->() {return &m_stats;}
    StorageStats& operator*() {return m_stats;}

    ~StorageStatsSentry() {m_mutex.unlock();}
  private:
    StorageStatsSentry() {m_mutex.lock();}
  };

  static StorageStatsSentry&& summaryLocked() {return std::move(StorageStatsSentry());}
  static const StorageStats& summary(void);
  static std::string         summaryText(bool banner=false);
  static void                fillSummary(std::map<std::string, std::string> &summary);
  static Counter&            counter (const std::string &storageClass,
                                      const std::string &operation);

private:
  static std::mutex m_mutex;
  static StorageStats m_stats;

};

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_H
