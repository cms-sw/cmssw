
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "QualityMetric.h"

using namespace XrdAdaptor;

QualityMetricWatch::QualityMetricWatch(QualityMetric *parent1, QualityMetric *parent2)
    : m_parent1(parent1), m_parent2(parent2)
{
    // TODO: just assuming success.
    clock_gettime(CLOCK_MONOTONIC, &m_start);
}

QualityMetricWatch::~QualityMetricWatch()
{
    if (m_parent1 && m_parent2)
    {
        timespec stop;
        clock_gettime(CLOCK_MONOTONIC, &stop);
        int ms = 1000*(stop.tv_sec - m_start.tv_sec) + (stop.tv_nsec - m_start.tv_nsec)/1e6;
        edm::LogVerbatim("XrdAdaptorInternal") << "Finished timer after " << ms << std::endl;
        m_parent1->finishWatch(stop, ms);
        m_parent2->finishWatch(stop, ms);
    }
}

QualityMetricWatch::QualityMetricWatch(QualityMetricWatch &&that)
{
    m_parent1 = that.m_parent1;
    m_parent2 = that.m_parent2;
    m_start = that.m_start;
    that.m_parent1 = nullptr;
    that.m_parent2 = nullptr;
    that.m_start = {0, 0};
}

void
QualityMetricWatch::swap(QualityMetricWatch &that)
{
    QualityMetric *tmp;
    tmp = that.m_parent1;
    that.m_parent1 = m_parent1;
    m_parent1 = tmp;
    tmp = that.m_parent2;
    that.m_parent2 = m_parent2;
    m_parent2 = tmp;
    timespec tmp2;
    tmp2 = that.m_start;
    that.m_start = m_start;
    m_start = tmp2;
}


QualityMetric::QualityMetric(timespec now, int default_value)
    : m_value(default_value),
      m_interval0_n(0),
      m_interval0_val(-1),
      m_interval0_start(now.tv_sec),
      m_interval1_val(-1),
      m_interval2_val(-1),
      m_interval3_val(-1),
      m_interval4_val(-1)
{
}

void
QualityMetric::finishWatch(timespec stop, int ms)
{
    m_value = -1;
    if (stop.tv_sec > m_interval0_start+interval_length)
    {
        m_interval4_val = m_interval3_val;
        m_interval3_val = m_interval2_val;
        m_interval2_val = m_interval1_val;
        m_interval1_val = m_interval0_val;
        m_interval0_n = 1;
        m_interval0_val = ms;
        m_interval0_start = stop.tv_sec;
    }
    else
    {
        int num = m_interval0_val * m_interval0_n + ms;
        m_interval0_n++;
        m_interval0_val = num / m_interval0_n;
    }
}

unsigned
QualityMetric::get()
{
    if (m_value == -1)
    {
        unsigned den = 0;
        m_value = 0;
        if (m_interval0_val >= 0)
        {
            den += 16;
            m_value = 16*m_interval0_val;
        }
        if (m_interval1_val >= 0)
        {
            den += 8;
            m_value += 8*m_interval1_val;
        }
        if (m_interval2_val >= 0)
        {
            den += 4;
            m_value += 4*m_interval2_val;
        }
        if (m_interval3_val >= 0)
        {
            den += 2;
            m_value += 2*m_interval3_val;
        }
        if (m_interval4_val >= 0)
        {
            den += 1;
            m_value += m_interval4_val;
        }
        if (den)
            m_value /= den;
        else
            m_value = 260;
    }
    return m_value;
}

QualityMetricFactory * QualityMetricFactory::m_instance = new QualityMetricFactory();

std::unique_ptr<QualityMetricSource>
QualityMetricFactory::get(timespec now, const std::string &id)
{
    MetricMap::const_iterator it = m_instance->m_sources.find(id);
    QualityMetricUniqueSource *source;
    if (it == m_instance->m_sources.end())
    {
        source = new QualityMetricUniqueSource(now);
        m_instance->m_sources[id] = source;
    }
    else
    {
        source = it->second;
    }
    return source->newSource(now);
}

QualityMetricSource::QualityMetricSource(QualityMetricUniqueSource &parent, timespec now, int default_value)
    : QualityMetric(now, default_value),
      m_parent(parent)
{}

void
QualityMetricSource::startWatch(QualityMetricWatch & watch)
{
    QualityMetricWatch tmp(&m_parent, this);
    watch.swap(tmp);
}

QualityMetricUniqueSource::QualityMetricUniqueSource(timespec now)
    : QualityMetric(now)
{}

std::unique_ptr<QualityMetricSource>
QualityMetricUniqueSource::newSource(timespec now)
{
    std::unique_ptr<QualityMetricSource> child(new QualityMetricSource(*this, now, get()));
    return child;
}

