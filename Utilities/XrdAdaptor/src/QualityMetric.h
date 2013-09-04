#ifndef Utilities_XrdAdaptor_QualityMetric_h
#define Utilities_XrdAdaptor_QualityMetric_h

#include <time.h>

#include <memory>
#include <unordered_map>

#include <boost/utility.hpp>

namespace XrdAdaptor {

class QualityMetric;
class QualityMetricSource;
class QualityMetricUniqueSource;

class QualityMetricWatch : boost::noncopyable {
friend class QualityMetricSource;

public:
    QualityMetricWatch() : m_parent1(nullptr), m_parent2(nullptr) {}
    QualityMetricWatch(QualityMetricWatch &&);
    ~QualityMetricWatch();

    void swap(QualityMetricWatch &);

private:
    QualityMetricWatch(QualityMetric *parent1, QualityMetric *parent2);
    timespec m_start;
    QualityMetric *m_parent1;
    QualityMetric *m_parent2;
};

class QualityMetric : boost::noncopyable {
friend class QualityMetricWatch;

public:
    QualityMetric(timespec now, int default_value=260);
    unsigned get();

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

};

class QualityMetricFactory {

friend class Source;

private:
    static
    std::unique_ptr<QualityMetricSource> get(timespec now, const std::string &id);

    static QualityMetricFactory *m_instance;

    typedef std::unordered_map<std::string, QualityMetricUniqueSource*> MetricMap;
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

}

#endif // Utilities_XrdAdaptor_QualityMetric_h

