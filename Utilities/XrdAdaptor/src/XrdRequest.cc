
#include <iostream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "XrdRequest.h"
#include "XrdRequestManager.h"

using namespace XrdAdaptor;

// If you define XRD_FAKE_ERROR, 1/5 read requests should fail.
#ifdef XRD_FAKE_ERROR
#define FAKE_ERROR_COUNTER 5
int g_fakeError = 0;
#else
#define FAKE_ERROR_COUNTER 0
int g_fakeError = 0;
#endif

XrdAdaptor::ClientRequest::~ClientRequest() {}

void 
XrdAdaptor::ClientRequest::HandleResponse(XrdCl::XRootDStatus *stat, XrdCl::AnyObject *resp)
{
    std::unique_ptr<XrdCl::AnyObject> response(resp);
    std::unique_ptr<XrdCl::XRootDStatus> status(stat);
    std::shared_ptr<ClientRequest> self_ref = m_self_reference;
    m_self_reference.reset();
    {
        QualityMetricWatch qmw;
        m_qmw.swap(qmw);
    }
    m_stats.reset();

    if ((!FAKE_ERROR_COUNTER || ((++g_fakeError % FAKE_ERROR_COUNTER) != 0)) && (status->IsOK() && resp))
    {
        if (m_into)
        {
            XrdCl::ChunkInfo *read_info;
            response->Get(read_info);
            m_promise.set_value(read_info->length);
        }
        else
        {
            XrdCl::VectorReadInfo *read_info;
            response->Get(read_info);
            m_promise.set_value(read_info->GetSize());
        }
    }
    else
    {
        Source *source = m_source.get();
        edm::LogWarning("XrdAdaptorInternal") << "XrdRequestManager::handle(name='"
          << m_manager.getFilename() << ") failure when reading from "
          << (source ? source->PrettyID() : "(unknown source)")
          << "; failed with error '" << status->ToStr() << "' (errno="
          << status->errNo << ", code=" << status->code << ").";
        m_failure_count++;
        try
        {
            try
            {
                m_manager.requestFailure(self_ref, *status);
                return;
            }
            catch (XrootdException& ex)
            {
                if (ex.getCode() == XrdCl::errInvalidResponse)
                {
                    m_promise.set_exception(std::current_exception());
                }
                else throw;
            }
        }
        catch (edm::Exception& ex)
        {
            ex.addContext("XrdAdaptor::ClientRequest::HandleResponse() failure while running connection recovery");
            std::stringstream ss;
            ss << "Original error: '" << status->ToStr() << "' (errno="
              << status->errNo << ", code=" << status->code << ", source=" << (source ? source->PrettyID() : "(unknown source)") << ").";
            ex.addAdditionalInfo(ss.str());
            m_promise.set_exception(std::current_exception());
            edm::LogWarning("XrdAdaptorInternal") << "Caught a CMSSW exception when running connection recovery.";
        }
        catch (...)
        {
            edm::Exception ex(edm::errors::FileReadError);
            ex << "XrdRequestManager::handle(name='" << m_manager.getFilename()
               << ") failed with error '" << status->ToStr()
               << "' (errno=" << status->errNo << ", code="
               << status->code << ").  Unknown exception occurred when running"
               << " connection recovery.";
            ex.addContext("Calling XrdRequestManager::handle()");
            m_manager.addConnections(ex);
            ex.addAdditionalInfo("Original source of error is " + (source ? source->PrettyID() : "(unknown source)"));
            m_promise.set_exception(std::make_exception_ptr(ex));
            edm::LogWarning("XrdAdaptorInternal") << "Caught a new exception when running connection recovery.";
        }
    }
}

