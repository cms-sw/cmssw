#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_H

# include <boost/shared_ptr.hpp>
# include <stdint.h>
# include <string>
# include <map>
# include <mutex>
# include <chrono>

class StorageAccount {
public:
  struct Counter {
    uint64_t attempts;
    uint64_t successes;
    uint64_t amount;
    // NOTE: Significant risk exists for underflow in this value.
    // However, the use cases are marginal so I let it pass.
    double   amount_square;
    int64_t  vector_count;
    int64_t  vector_square;
    double   timeTotal;
    double   timeMin;
    double   timeMax;
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
  typedef std::map<std::string, boost::shared_ptr<OperationStats> > StorageStats;

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
  static std::string         summaryXML(void);
  static std::string         summaryText(bool banner=false);
  static void                fillSummary(std::map<std::string, std::string> &summary);
  static Counter&            counter (const std::string &storageClass,
                                      const std::string &operation);

private:
  static std::mutex m_mutex;
  static StorageStats m_stats;

};

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_H
