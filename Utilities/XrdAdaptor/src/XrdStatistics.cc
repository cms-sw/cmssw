
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"

#include "XrdRequest.h"
#include "XrdStatistics.h"

#include <chrono>

using namespace XrdAdaptor;


std::atomic<XrdSiteStatisticsInformation*> XrdSiteStatisticsInformation::m_instance;


XrdStatisticsService::XrdStatisticsService(const edm::ParameterSet &iPS, edm::ActivityRegistry &iRegistry)
{
    XrdSiteStatisticsInformation::createInstance();

    if (iPS.getUntrackedParameter<bool>("reportToFJR", false))
    {
        iRegistry.watchPostEndJob(this, &XrdStatisticsService::postEndJob);
    }
}


void XrdStatisticsService::postEndJob()
{
    edm::Service<edm::JobReport> reportSvc;
    if (!reportSvc.isAvailable()) {return;}

    XrdSiteStatisticsInformation *instance = XrdSiteStatisticsInformation::getInstance();
    if (!instance) {return;}

    std::map<std::string, std::string> props;
    for (auto& stats : instance->m_sites)
    {
        stats->recomputeProperties(props);
        reportSvc->reportPerformanceForModule(stats->site(), "XrdSiteStatistics", props);
    }
}

std::vector<std::pair<std::string, XrdStatisticsService::CondorIOStats>>
XrdStatisticsService::condorUpdate()
{
    std::vector<std::pair<std::string, XrdStatisticsService::CondorIOStats>> result;
    XrdSiteStatisticsInformation *instance = XrdSiteStatisticsInformation::getInstance();
    if (!instance) {return result;}

    std::lock_guard<std::mutex> lock(instance->m_mutex);
    result.reserve(instance->m_sites.size());
    for (auto& stats : instance->m_sites)
    {
        CondorIOStats cs;
        std::shared_ptr<XrdSiteStatistics> ss = get_underlying_safe(stats);
        if (!ss) continue;
        cs.bytesRead = ss->getTotalBytes();
        cs.transferTime = ss->getTotalReadTime();
        result.emplace_back(ss->site(), cs);
    }
    return result;
}


std::shared_ptr<XrdSiteStatistics>
XrdSiteStatisticsInformation::getStatisticsForSite(std::string const &site)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto& stats : m_sites)
    {
        if (stats->site() == site) {return get_underlying_safe(stats);}
    }
    m_sites.emplace_back(new XrdSiteStatistics(site));
    return get_underlying_safe(m_sites.back());
}


void
XrdSiteStatisticsInformation::createInstance()
{
    if (!m_instance)
    {
        std::unique_ptr<XrdSiteStatisticsInformation> tmp { new XrdSiteStatisticsInformation() };
        XrdSiteStatisticsInformation* expected = nullptr;
        if (m_instance.compare_exchange_strong(expected,tmp.get()))
        {
            tmp.release();
        }
    }
}

XrdSiteStatisticsInformation *
XrdSiteStatisticsInformation::getInstance()
{
    return m_instance.load(std::memory_order_relaxed);
}

void XrdStatisticsService::fillDescriptions(edm::ConfigurationDescriptions &descriptions)
{
    edm::ParameterSetDescription desc;
    desc.setComment("Report Xrootd-related statistics centrally.");
    desc.addUntracked<bool>("reportToFJR", true)
        ->setComment("True: Add per-site Xrootd statistics to the framework job report.\n"
                     "False: Collect no site-specific statistics.\n");
    descriptions.add("XrdAdaptor::XrdStatisticsService", desc);
}


XrdSiteStatistics::XrdSiteStatistics(std::string const &site) :
    m_site(site),
    m_readvCount(0),
    m_chunkCount(0),
    m_readvSize(0),
    m_readvNS(0.0),
    m_readCount(0),
    m_readSize(0),
    m_readNS(0)
{
}

std::shared_ptr<XrdReadStatistics>
XrdSiteStatistics::startRead(std::shared_ptr<XrdSiteStatistics> parent, std::shared_ptr<ClientRequest> req)
{
    std::shared_ptr<XrdReadStatistics> readStats(new XrdReadStatistics(parent, req->getSize(), req->getCount()));
    return readStats;
}


static std::string
i2str(int input)
{
    std::ostringstream formatter;
    formatter << input;
    return formatter.str();
}


static std::string
d2str(double input)
{
    std::ostringstream formatter;
    formatter << std::setw(4) << input;
    return formatter.str();
}


void
XrdSiteStatistics::recomputeProperties(std::map<std::string, std::string> &props)
{
    props.clear();

    props["readv-numOperations"] = i2str(m_readvCount);
    props["readv-numChunks"] = i2str(m_chunkCount);
    props["readv-totalMegabytes"] = d2str(static_cast<float>(m_readvSize)/(1024.0*1024.0));
    props["readv-totalMsecs"] = d2str(m_readvNS/1e6);

    props["read-numOperations"] = i2str(m_readCount);
    props["read-totalMegabytes"] = d2str(static_cast<float>(m_readSize)/(1024.0*1024.0));
    props["read-totalMsecs"] = d2str(static_cast<float>(m_readNS)/1e6);
}


void
XrdSiteStatistics::finishRead(XrdReadStatistics const &readStats)
{
    if (readStats.readCount() > 1)
    {
        m_readvCount ++;
        m_chunkCount += readStats.readCount();
        m_readvSize += readStats.size();
        m_readvNS += readStats.elapsedNS();
    }
    else
    {
        m_readCount ++;
        m_readSize += readStats.size();
        m_readNS += readStats.elapsedNS();
    }
}


XrdReadStatistics::XrdReadStatistics(std::shared_ptr<XrdSiteStatistics> parent, IOSize size, size_t count) :
    m_size(size),
    m_count(count),
    m_parent(parent),
    m_start(std::chrono::high_resolution_clock::now())
{
}


uint64_t
XrdReadStatistics::elapsedNS() const
{
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end-m_start).count();
}

