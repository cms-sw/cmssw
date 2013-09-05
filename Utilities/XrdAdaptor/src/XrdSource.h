#ifndef Utilities_XrdAdaptor_XrdSource_h
#define Utilities_XrdAdaptor_XrdSource_h

#include <memory>
#include <vector>

#include <boost/utility.hpp>

#include "QualityMetric.h"

namespace XrdCl {
    class File;
}

namespace XrdAdaptor {

class RequestList;
class ClientRequest;

class Source : public std::enable_shared_from_this<Source>, boost::noncopyable {

public:
    Source(timespec now, std::unique_ptr<XrdCl::File> fileHandle);

    ~Source();

    void handle(std::shared_ptr<ClientRequest>);

    void handle(RequestList &);

    std::shared_ptr<XrdCl::File> getFileHandle();

    const std::string & ID() const {return m_id;}

    unsigned getQuality() {return m_qm->get();}

    struct timespec getLastDowngrade() const {return m_lastDowngrade;}
    void setLastDowngrade(struct timespec now) {m_lastDowngrade = now;}

private:
    void requestCallback(/* TODO: type? */);

    struct timespec m_lastDowngrade;
    std::string m_id;
    std::shared_ptr<XrdCl::File> m_fh;

    std::unique_ptr<QualityMetricSource> m_qm;

    std::vector<char> m_buffer;

#ifdef XRD_FAKE_SLOW
    bool m_slow;
#endif
};

}

#endif
