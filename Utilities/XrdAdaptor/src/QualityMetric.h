#ifndef Utilities_XrdAdaptor_QualityMetric_h
#define Utilities_XrdAdaptor_QualityMetric_h

#include <ctime>

#include <mutex>
#include <memory>

#include "tbb/concurrent_unordered_map.h"

#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

namespace XrdAdaptor {

  class QualityMetric;
  class QualityMetricSource;
  class QualityMetricUniqueSource;

  class QualityMetricWatch {
    friend class QualityMetricSource;

  public:
    QualityMetricWatch() : m_parent1(nullptr), m_parent2(nullptr) {}
    QualityMetricWatch(QualityMetricWatch &&);
    ~QualityMetricWatch();

    QualityMetricWatch(const QualityMetricWatch &) = delete;
    QualityMetricWatch &operator=(const QualityMetricWatch &) = delete;

    void swap(QualityMetricWatch &);

  private:
    QualityMetricWatch(QualityMetric *parent1, QualityMetric *parent2);
    timespec m_start;
    edm::propagate_const<QualityMetric *> m_parent1;
    edm::propagate_const<QualityMetric *> m_parent2;
  };

  class QualityMetric {
    friend class QualityMetricWatch;

  public:
    QualityMetric(timespec now, int default_value = 260);
    unsigned get();

    QualityMetric(const QualityMetric &) = delete;
    QualityMetric &operator=(const QualityMetric &) = delete;

  private:
    void finishWatch(timespec now, int ms);

    static const unsigned interval_length = 60;

    int m_value;
    int m_interval0_n;
    int m_interval0_val;
    time_t m_interval0_start;
    int m_interval1_val;
    int m_interval2_val;
    int m_interval3_val;
    int m_interval4_val;

    std::mutex m_mutex;
  };

  class QualityMetricFactory {
    friend class Source;

  private:
    static std::unique_ptr<QualityMetricSource> get(timespec now, const std::string &id);

    CMS_THREAD_SAFE static QualityMetricFactory m_instance;

    typedef tbb::concurrent_unordered_map<std::string, QualityMetricUniqueSource *> MetricMap;
    MetricMap m_sources;
  };

  /**
 * This QM implementation is meant to be held by each XrdAdaptor::Source
 * instance
 */
  class QualityMetricSource final : public QualityMetric {
    friend class QualityMetricUniqueSource;

  public:
    void startWatch(QualityMetricWatch &);

  private:
    QualityMetricSource(QualityMetricUniqueSource &parent, timespec now, int default_value);

    QualityMetricUniqueSource &m_parent;
  };

  /*
 * This quality metric tracks all accesses to a given source ID.
 */
  class QualityMetricUniqueSource final : public QualityMetric {
    friend class QualityMetricFactory;

  private:
    QualityMetricUniqueSource(timespec now);
    std::unique_ptr<QualityMetricSource> newSource(timespec now);
  };

}  // namespace XrdAdaptor

#endif  // Utilities_XrdAdaptor_QualityMetric_h
