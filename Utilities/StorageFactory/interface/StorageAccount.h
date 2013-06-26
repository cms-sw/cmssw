#ifndef STORAGE_FACTORY_STORAGE_ACCOUNT_H
# define STORAGE_FACTORY_STORAGE_ACCOUNT_H

# include <boost/shared_ptr.hpp>
# include <stdint.h>
# include <string>
# include <map>

class StorageAccount {
public:
  struct Counter {
    uint64_t attempts;
    uint64_t successes;
    double   amount;
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

    void     tick (double amount = 0., int64_t tick = 0) const;
  protected:
    Counter &m_counter;
    double   m_start;
  };

  typedef std::map<std::string, Counter> OperationStats;
  typedef std::map<std::string, boost::shared_ptr<OperationStats> > StorageStats;

  static const StorageStats& summary(void);
  static std::string         summaryXML(void);
  static std::string         summaryText(bool banner=false);
  static void                fillSummary(std::map<std::string, std::string> &summary);
  static Counter&            counter (const std::string &storageClass,
                                      const std::string &operation);
};

#endif // STORAGE_FACTORY_STORAGE_ACCOUNT_H
