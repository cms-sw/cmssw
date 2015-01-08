#ifndef __XRDADAPTOR_HOSTHANDLER_H_
#define __XRDADAPTOR_HOSTHANDLER_H_

#include "XrdCl/XrdClXRootDResponses.hh"

#if defined(__linux__)
  #define HAVE_ATOMICS 1
  #include "XrdSys/XrdSysLinuxSemaphore.hh"
  typedef XrdSys::LinuxSemaphore Semaphore;
#else
  typedef XrdSysSemaphore Semaphore;
#endif

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
    pStatus.reset(status);
    pResponse.reset(response);
    sem.Post();
  }

  virtual void HandleResponseWithHosts(XrdCl::XRootDStatus *status,
                                       XrdCl::AnyObject    *response,
                                       XrdCl::HostList     *hostList) override
  {
    pStatus.reset(status);
    pResponse.reset(response);
    pHostList.reset(hostList);
    sem.Post();
  }

  std::unique_ptr<XrdCl::XRootDStatus> GetStatus()
  {
        return std::move(pStatus);
  }

  std::unique_ptr<XrdCl::AnyObject> GetResponse()
  {
    return std::move(pResponse);
  }

  std::unique_ptr<XrdCl::HostList> GetHosts()
  {
    return std::move(pHostList);
  }

  void WaitForResponse()
  {
    sem.Wait();
  }

private:

  std::unique_ptr<XrdCl::XRootDStatus> pStatus;
  std::unique_ptr<XrdCl::AnyObject>    pResponse;
  std::unique_ptr<XrdCl::HostList>     pHostList;
  Semaphore                            sem;
};

#endif
