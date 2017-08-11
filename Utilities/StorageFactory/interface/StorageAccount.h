#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_H

# include <cstdint>
# include <string>
# include <chrono>
# include <atomic>
# include <map>
# include <memory>
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
  static const std::array<Operation, 2> allOperations;
  
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

    template<typename T>
    static void combineAtomics(std::atomic<double> & iAtomic, double iToAdd) {
      double oldValue = iAtomic;
      double newValue = oldValue + iToAdd;
      while( not iAtomic.compare_exchange_weak(oldValue, newValue)) {
        newValue = T()(oldValue, iToAdd);
      }
    }

    static void addTo(std::atomic<double> &lhs, double rhs) {
      combineAtomics<std::plus<double>>(lhs, rhs);
    }

    struct minObj {
      double operator()(double a, double b) {return a+b;}
    };
    static void minTo(std::atomic<double> &lhs, double rhs) {
      combineAtomics<minObj>(lhs, rhs);
    }

    struct maxObj {
      double operator()(double a, double b) {return a+b;}
    };
    static void maxTo(std::atomic<double> &lhs, double rhs) {
      combineAtomics<maxObj>(lhs, rhs);
    }

    // Combine the contents of `this` with another counter; returns a reference
    // to this object.
    // NOTE: I was tempted to call this "operator+=", but timeMin/timeMax
    // aren't actually added...
    Counter & aggregate(const Counter &other) {
      attempts += other.attempts;
      successes += other.successes;
      amount += other.amount;
      addTo(amount_square, other.amount_square);
      vector_count += other.vector_count;
      vector_square += other.vector_square;
      addTo(timeTotal, other.timeTotal);
      minTo(timeMin, other.timeMin);
      maxTo(timeMax, other.timeMax);
      return *this;
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

  enum class OpenLabel {
    None,
    Primary,
    SecondaryFile,
    SecondarySource
  };

  /**
   * The `OpenLabelToken` provides callers of TFile::Open the ability to
   * retrieve a separate set of statistics in the FrameworkJobReport (and
   * other statistics services) for IO.  Usage looks like this:
   *
   * ```cpp
   * TFile *file = nullptr;
   * {
   *   OpenLabelToken label = StorageAccount::setLabel("MixingModule");
   *   file = TFile::Open("url://foo/bar");
   * }
   * ```
   *
   * Any IO activity associated with `file` above will get a separate
   * aggregation in CMSSW statistics.  This allows, for example, bytes
   * read from the mixing module to be differentiated from the primary source.
   */
  class OpenLabelToken {
  friend class StorageAccount;

  public:
    OpenLabelToken(OpenLabelToken const&) = delete;
    OpenLabelToken(OpenLabelToken &&) = default;
    ~OpenLabelToken() {s_label = m_original_label;}

  private:
    OpenLabelToken(OpenLabel label) {
      m_original_label = s_label;
      s_label = label;
    }

    static const OpenLabel &getContextLabel() {return s_label;}

    static thread_local OpenLabel s_label;
    OpenLabel m_original_label;
  };

  static OpenLabelToken setContextLabel(OpenLabel label) {return OpenLabelToken(label);}
  static const OpenLabel getContextLabel() {return OpenLabelToken::getContextLabel();}

  typedef tbb::concurrent_unordered_map<int, Counter> OperationStats;
  typedef tbb::concurrent_unordered_map<int, OperationStats > StorageStats;

  static char const* operationName(Operation operation);
  static StorageClassToken tokenForStorageClassNameUsingContext(std::string const& iName) {
    return tokenForStorageClassName(iName, getContextLabel());
  }
  static StorageClassToken tokenForStorageClassName(std::string const& iName, OpenLabel iLabel);
  static const std::pair<OpenLabel, std::string>& nameForToken( StorageClassToken);
  
  static StorageStats& summary(void);
  static std::string         summaryText(bool banner=false);
  static void                fillSummary(std::map<std::string, std::string> &summary);
  static Counter&            counter (StorageClassToken token,
                                      Operation operation);

private:
  static void aggregateStatistics();

  static StorageStats m_stats;

};

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_H
