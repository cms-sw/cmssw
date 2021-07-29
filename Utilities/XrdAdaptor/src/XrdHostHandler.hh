#ifndef __XRDADAPTOR_HOSTHANDLER_H_
#define __XRDADAPTOR_HOSTHANDLER_H_

#include "XrdCl/XrdClXRootDResponses.hh"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

/**
 * The SyncResponseHandler from the XrdCl does not
 * preserve the hostinfo list, which we would like to
 * utilize.
 */

class SyncHostResponseHandler: public XrdCl::ResponseHandler
{
public:

  SyncHostResponseHandler():
    sem(0)
  {
  }

  virtual ~SyncHostResponseHandler() = default;


  virtual void HandleResponse(XrdCl::XRootDStatus *status,
                              XrdCl::AnyObject    *response) override
  {
    // propagate_const<T> has no reset() function
    pStatus_ = std::unique_ptr<XrdCl::XRootDStatus>(status);
    pResponse_ = std::unique_ptr<XrdCl::AnyObject>(response);
    sem.Post();
  }

  virtual void HandleResponseWithHosts(XrdCl::XRootDStatus *status,
                                       XrdCl::AnyObject    *response,
                                       XrdCl::HostList     *hostList) override
  {
    // propagate_const<T> has no reset() function
    pStatus_ = std::unique_ptr<XrdCl::XRootDStatus>(status);
    pResponse_ = std::unique_ptr<XrdCl::AnyObject>(response);
    pHostList_ = std::unique_ptr<XrdCl::HostList>(hostList);
    sem.Post();
  }

  std::unique_ptr<XrdCl::XRootDStatus> GetStatus()
  {
        return std::move(get_underlying_safe(pStatus_));
  }

  std::unique_ptr<XrdCl::AnyObject> GetResponse()
  {
    return std::move(get_underlying_safe(pResponse_));
  }

  std::unique_ptr<XrdCl::HostList> GetHosts()
  {
    return std::move(get_underlying_safe(pHostList_));
  }

  void WaitForResponse()
  {
    sem.Wait();
  }

private:

  edm::propagate_const<std::unique_ptr<XrdCl::XRootDStatus>> pStatus_;
  edm::propagate_const<std::unique_ptr<XrdCl::AnyObject>> pResponse_;
  edm::propagate_const<std::unique_ptr<XrdCl::HostList>> pHostList_;
  XrdSysSemaphore sem;
};

#endif
