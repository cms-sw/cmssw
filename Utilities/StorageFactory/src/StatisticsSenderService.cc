
#include "Utilities/StorageFactory/interface/StatisticsSenderService.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/src/Guid.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include <string>
#include <cmath>

#include <unistd.h>
#include <fcntl.h>

#include <openssl/x509.h>
#include <openssl/pem.h>

#define OUTPUT_STATISTIC(x) os << "\"" #x "\":" << (x - m_##x) << ", ";

// Simple hack to define HOST_NAME_MAX on Mac.
// Allows arrays to be statically allocated
#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 128
#endif

static constexpr char const *const JOB_UNIQUE_ID_ENV = "CRAB_UNIQUE_JOB_ID";
static constexpr char const *const JOB_UNIQUE_ID_ENV_V2 = "DashboardJobId";

using namespace edm::storage;

StatisticsSenderService::FileStatistics::FileStatistics()
    : m_read_single_operations(0),
      m_read_single_bytes(0),
      m_read_single_square(0),
      m_read_vector_operations(0),
      m_read_vector_bytes(0),
      m_read_vector_square(0),
      m_read_vector_count_sum(0),
      m_read_vector_count_square(0),
      m_start_time(time(nullptr)) {}

void StatisticsSenderService::FileStatistics::fillUDP(std::ostringstream &os) const {
  const StorageAccount::StorageStats &stats = StorageAccount::summary();
  ssize_t read_single_operations = 0;
  ssize_t read_single_bytes = 0;
  ssize_t read_single_square = 0;
  ssize_t read_vector_operations = 0;
  ssize_t read_vector_bytes = 0;
  ssize_t read_vector_square = 0;
  ssize_t read_vector_count_sum = 0;
  ssize_t read_vector_count_square = 0;
  auto token = StorageAccount::tokenForStorageClassName("tstoragefile");
  for (StorageAccount::StorageStats::const_iterator i = stats.begin(); i != stats.end(); ++i) {
    if (i->first == token.value()) {
      continue;
    }
    for (StorageAccount::OperationStats::const_iterator j = i->second.begin(); j != i->second.end(); ++j) {
      if (j->first == static_cast<int>(StorageAccount::Operation::readv)) {
        read_vector_operations += j->second.attempts;
        read_vector_bytes += j->second.amount;
        read_vector_count_square += j->second.vector_square;
        read_vector_square += j->second.amount_square;
        read_vector_count_sum += j->second.vector_count;
      } else if (j->first == static_cast<int>(StorageAccount::Operation::read)) {
        read_single_operations += j->second.attempts;
        read_single_bytes += j->second.amount;
        read_single_square += j->second.amount_square;
      }
    }
  }
  int64_t single_op_count = read_single_operations - m_read_single_operations;
  if (single_op_count > 0) {
    double single_sum = read_single_bytes - m_read_single_bytes;
    double single_average = single_sum / static_cast<double>(single_op_count);
    os << "\"read_single_sigma\":"
       << sqrt(std::abs((static_cast<double>(read_single_square - m_read_single_square) -
                         single_average * single_average * single_op_count) /
                        static_cast<double>(single_op_count)))
       << ", ";
    os << "\"read_single_average\":" << single_average << ", ";
  }
  int64_t vector_op_count = read_vector_operations - m_read_vector_operations;
  if (vector_op_count > 0) {
    double vector_average =
        static_cast<double>(read_vector_bytes - m_read_vector_bytes) / static_cast<double>(vector_op_count);
    os << "\"read_vector_average\":" << vector_average << ", ";
    os << "\"read_vector_sigma\":"
       << sqrt(std::abs((static_cast<double>(read_vector_square - m_read_vector_square) -
                         vector_average * vector_average * vector_op_count) /
                        static_cast<double>(vector_op_count)))
       << ", ";
    double vector_count_average =
        static_cast<double>(read_vector_count_sum - m_read_vector_count_sum) / static_cast<double>(vector_op_count);
    os << "\"read_vector_count_average\":" << vector_count_average << ", ";
    os << "\"read_vector_count_sigma\":"
       << sqrt(std::abs((static_cast<double>(read_vector_count_square - m_read_vector_count_square) -
                         vector_count_average * vector_count_average * vector_op_count) /
                        static_cast<double>(vector_op_count)))
       << ", ";
  }

  os << "\"read_bytes\":" << (read_vector_bytes + read_single_bytes - m_read_vector_bytes - m_read_single_bytes)
     << ", ";
  os << "\"read_bytes_at_close\":"
     << (read_vector_bytes + read_single_bytes - m_read_vector_bytes - m_read_single_bytes) << ", ";

  // See top of file for macros; not complex, just avoiding copy/paste
  OUTPUT_STATISTIC(read_single_operations)
  OUTPUT_STATISTIC(read_single_bytes)
  OUTPUT_STATISTIC(read_vector_operations)
  OUTPUT_STATISTIC(read_vector_bytes)

  os << "\"start_time\":" << m_start_time << ", ";
  // NOTE: last entry doesn't have the trailing comma.
  os << "\"end_time\":" << time(nullptr);
}

void StatisticsSenderService::FileStatistics::update() {
  const StorageAccount::StorageStats &stats = StorageAccount::summary();
  ssize_t read_single_operations = 0;
  ssize_t read_single_bytes = 0;
  ssize_t read_single_square = 0;
  ssize_t read_vector_operations = 0;
  ssize_t read_vector_bytes = 0;
  ssize_t read_vector_square = 0;
  ssize_t read_vector_count_sum = 0;
  ssize_t read_vector_count_square = 0;
  auto token = StorageAccount::tokenForStorageClassName("tstoragefile");
  for (StorageAccount::StorageStats::const_iterator i = stats.begin(); i != stats.end(); ++i) {
    if (i->first == token.value()) {
      continue;
    }
    for (StorageAccount::OperationStats::const_iterator j = i->second.begin(); j != i->second.end(); ++j) {
      if (j->first == static_cast<int>(StorageAccount::Operation::readv)) {
        read_vector_operations += j->second.attempts;
        read_vector_bytes += j->second.amount;
        read_vector_count_square += j->second.vector_square;
        read_vector_square += j->second.amount_square;
        read_vector_count_sum += j->second.vector_count;
      } else if (j->first == static_cast<int>(StorageAccount::Operation::read)) {
        read_single_operations += j->second.attempts;
        read_single_bytes += j->second.amount;
        read_single_square += j->second.amount_square;
      }
    }
  }

  m_read_single_square = read_single_square;
  m_read_vector_square = read_vector_square;
  m_read_vector_count_square = read_vector_count_square;
  m_read_vector_count_sum = read_vector_count_sum;
  m_read_single_operations = read_single_operations;
  m_read_single_bytes = read_single_bytes;
  m_read_vector_operations = read_vector_operations;
  m_read_vector_bytes = read_vector_bytes;
  m_start_time = time(nullptr);
}
StatisticsSenderService::FileInfo::FileInfo(std::string const &iLFN, edm::InputType iType)
    : m_filelfn(iLFN),
      m_serverhost("unknown"),
      m_serverdomain("unknown"),
      m_type(iType),
      m_size(-1),
      m_id(0),
      m_openCount(1) {}

StatisticsSenderService::StatisticsSenderService(edm::ParameterSet const &iPSet, edm::ActivityRegistry &ar)
    : m_clienthost("unknown"),
      m_clientdomain("unknown"),
      m_filestats(),
      m_guid(Guid().toString()),
      m_counter(0),
      m_userdn("unknown"),
      m_debug(iPSet.getUntrackedParameter<bool>("debug", false)) {
  determineHostnames();
  ar.watchPostCloseFile(this, &StatisticsSenderService::filePostCloseEvent);
  if (!getX509Subject(m_userdn)) {
    m_userdn = "unknown";
  }
}

const char *StatisticsSenderService::getJobID() {
  const char *id = getenv(JOB_UNIQUE_ID_ENV);
  // Dashboard developers requested that we migrate to this environment variable.
  return id ? id : getenv(JOB_UNIQUE_ID_ENV_V2);
}

std::string const *StatisticsSenderService::matchedLfn(std::string const &iURL) {
  auto found = m_urlToLfn.find(iURL);
  if (found != m_urlToLfn.end()) {
    return &found->second;
  }
  for (auto const &v : m_lfnToFileInfo) {
    if (v.first.size() < iURL.size()) {
      if (v.first == iURL.substr(iURL.size() - v.first.size())) {
        m_urlToLfn.emplace(iURL, v.first);
        return &m_urlToLfn.find(iURL)->second;
      }
    }
  }
  //does the lfn have a protocol and the iURL not?
  if (std::string::npos == iURL.find(':')) {
    for (auto const &v : m_lfnToFileInfo) {
      if ((std::string::npos != v.first.find(':')) and (v.first.size() > iURL.size())) {
        if (iURL == v.first.substr(v.first.size() - iURL.size())) {
          m_urlToLfn.emplace(iURL, v.first);
          return &m_urlToLfn.find(iURL)->second;
        }
      }
    }
  }

  return nullptr;
}

void StatisticsSenderService::setCurrentServer(const std::string &url, const std::string &servername) {
  size_t dot_pos = servername.find('.');
  std::string serverhost;
  std::string serverdomain;
  if (dot_pos == std::string::npos) {
    serverhost = servername.substr(0, servername.find(":"));
    serverdomain = "unknown";
  } else {
    serverhost = servername.substr(0, dot_pos);
    serverdomain = servername.substr(dot_pos + 1, servername.find(":") - dot_pos - 1);
    if (serverdomain.empty()) {
      serverdomain = "unknown";
    }
  }
  {
    auto lfn = matchedLfn(url);
    std::lock_guard<std::mutex> sentry(m_servermutex);
    if (nullptr != lfn) {
      auto found = m_lfnToFileInfo.find(*lfn);
      if (found != m_lfnToFileInfo.end()) {
        found->second.m_serverhost = std::move(serverhost);
        found->second.m_serverdomain = std::move(serverdomain);
      }
    } else if (m_debug) {
      edm::LogWarning("StatisticsSenderService") << "setCurrentServer: unknown url name " << url << "\n";
    }
  }
}

void StatisticsSenderService::openingFile(std::string const &lfn, edm::InputType type, size_t size) {
  m_urlToLfn.emplace(lfn, lfn);
  auto attempt = m_lfnToFileInfo.emplace(lfn, FileInfo{lfn, type});
  if (attempt.second) {
    attempt.first->second.m_size = size;
    attempt.first->second.m_id = m_counter++;
    edm::LogInfo("StatisticsSenderService") << "openingFile: opening " << lfn << "\n";
  } else {
    ++(attempt.first->second.m_openCount);
    edm::LogInfo("StatisticsSenderService") << "openingFile: re-opening" << lfn << "\n";
  }
}

void StatisticsSenderService::closedFile(std::string const &url, bool usedFallback) {
  edm::Service<edm::SiteLocalConfig> pSLC;
  if (!pSLC.isAvailable()) {
    return;
  }

  const struct addrinfo *addresses = pSLC->statisticsDestination();
  if (!addresses and !m_debug) {
    return;
  }

  std::set<std::string> const *info = pSLC->statisticsInfo();
  if (info && !info->empty() && (m_userdn != "unknown") &&
      ((info->find("dn") == info->end()) || (info->find("nodn") != info->end()))) {
    m_userdn = "not reported";
  }

  auto lfn = matchedLfn(url);
  if (nullptr != lfn) {
    auto found = m_lfnToFileInfo.find(*lfn);
    assert(found != m_lfnToFileInfo.end());

    std::string results;
    fillUDP(pSLC->siteName(), found->second, usedFallback, results);
    if (m_debug) {
      edm::LogSystem("StatisticSenderService") << "\n" << results << "\n";
    }

    for (const struct addrinfo *address = addresses; address != nullptr; address = address->ai_next) {
      int sock = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
      if (sock < 0) {
        continue;
      }
      auto close_del = [](int *iSocket) { close(*iSocket); };
      std::unique_ptr<int, decltype(close_del)> guard(&sock, close_del);
      if (sendto(sock, results.c_str(), results.size(), 0, address->ai_addr, address->ai_addrlen) >= 0) {
        break;
      }
    }

    auto c = --found->second.m_openCount;
    if (m_debug) {
      if (c == 0) {
        edm::LogWarning("StatisticsSenderService") << "fully closed: " << *lfn << "\n";
      } else {
        edm::LogWarning("StatisticsSenderService") << "partially closed: " << *lfn << "\n";
      }
    }
  } else if (m_debug) {
    edm::LogWarning("StatisticsSenderService") << "closed: unknown url name " << url << "\n";
  }
}

void StatisticsSenderService::cleanupOldFiles() {
  //remove entries with openCount of 0
  bool moreToTest = false;
  do {
    moreToTest = false;
    for (auto it = m_lfnToFileInfo.begin(); it != m_lfnToFileInfo.end(); ++it) {
      if (it->second.m_openCount == 0) {
        auto lfn = it->first;
        bool moreToTest2 = false;
        do {
          moreToTest2 = false;
          for (auto it2 = m_urlToLfn.begin(); it2 != m_urlToLfn.end(); ++it2) {
            if (it2->second == lfn) {
              m_urlToLfn.unsafe_erase(it2);
              moreToTest2 = true;
              break;
            }
          }
        } while (moreToTest2);

        m_lfnToFileInfo.unsafe_erase(it);
        moreToTest = true;
        break;
      }
    }
  } while (moreToTest);
}

void StatisticsSenderService::setSize(const std::string &url, size_t size) {
  auto lfn = matchedLfn(url);
  if (nullptr != lfn) {
    auto itFound = m_lfnToFileInfo.find(*lfn);
    if (itFound != m_lfnToFileInfo.end()) {
      itFound->second.m_size = size;
    }
  } else if (m_debug) {
    edm::LogWarning("StatisticsSenderService") << "setSize: unknown url name " << url << "\n";
  }
}

void StatisticsSenderService::filePostCloseEvent(std::string const &lfn, bool usedFallback) {
  //we are at a sync point in the framwework so no new files are being opened
  cleanupOldFiles();
  m_filestats.update();
}

void StatisticsSenderService::determineHostnames(void) {
  char tmpName[HOST_NAME_MAX];
  if (gethostname(tmpName, HOST_NAME_MAX) != 0) {
    // Sigh, no way to log errors from here.
    m_clienthost = "unknown";
  } else {
    m_clienthost = tmpName;
  }
  size_t dot_pos = m_clienthost.find(".");
  if (dot_pos == std::string::npos) {
    m_clientdomain = "unknown";
  } else {
    m_clientdomain = m_clienthost.substr(dot_pos + 1, m_clienthost.size() - dot_pos - 1);
    m_clienthost = m_clienthost.substr(0, dot_pos);
  }
}

void StatisticsSenderService::fillUDP(const std::string &siteName,
                                      const FileInfo &fileinfo,
                                      bool usedFallback,
                                      std::string &udpinfo) const {
  std::ostringstream os;

  // Header - same for all IO accesses
  os << "{";
  if (!siteName.empty()) {
    os << "\"site_name\":\"" << siteName << "\", ";
  }
  // edm::getReleaseVersion() returns a string that includes quotation
  // marks, therefore they are not added here
  os << "\"cmssw_version\":" << edm::getReleaseVersion() << ", ";
  if (usedFallback) {
    os << "\"fallback\": true, ";
  } else {
    os << "\"fallback\": false, ";
  }
  os << "\"read_type\": ";
  switch (fileinfo.m_type) {
    case edm::InputType::Primary: {
      os << "\"primary\", ";
      break;
    }
    case edm::InputType::SecondaryFile: {
      os << "\"secondary\", ";
      break;
    }
    case edm::InputType::SecondarySource: {
      os << "\"embedded\", ";
      break;
    }
  }
  auto serverhost = fileinfo.m_serverhost;
  auto serverdomain = fileinfo.m_serverdomain;

  os << "\"user_dn\":\"" << m_userdn << "\", ";
  os << "\"client_host\":\"" << m_clienthost << "\", ";
  os << "\"client_domain\":\"" << m_clientdomain << "\", ";
  os << "\"server_host\":\"" << serverhost << "\", ";
  os << "\"server_domain\":\"" << serverdomain << "\", ";
  os << "\"unique_id\":\"" << m_guid << "-" << fileinfo.m_id << "\", ";
  os << "\"file_lfn\":\"" << fileinfo.m_filelfn << "\", ";
  // Dashboard devs requested that we send out no app_info if a job ID
  // is not present in the environment.
  const char *jobId = getJobID();
  if (jobId) {
    os << "\"app_info\":\"" << jobId << "\", ";
  }

  if (fileinfo.m_size >= 0) {
    os << "\"file_size\":" << fileinfo.m_size << ", ";
  }

  m_filestats.fillUDP(os);

  os << "}";
  udpinfo = os.str();
}

/*
 * Pull the X509 user subject from the environment.
 * Based on initial code from the Frontier client:
 *   http://cdcvs.fnal.gov/cgi-bin/public-cvs/cvsweb-public.cgi/~checkout~/frontier/client/frontier.c?rev=1.57&content-type=text/plain
 * This was further extended by walking up the returned chain similar to the Globus function
 *   globus_gsi_cert_utils-6.6/library/globus_gsi_cert_utils.c:globus_gsi_cert_utils_get_eec
 *   globus_gsi_credential-3.5/library/globus_gsi_credential.c:globus_gsi_cred_read_proxy_bio
 */

/* 
 * Given a stack of x509 proxies, take a guess at the EEC.
 * Assumes the proxies are in reverse sorted order and looks for the first
 * proxy which is not a substring of the prior proxy.
 * THIS DOES NOT VERIFY THE RESULTS, and is a best-effort GUESS.
 * Again, DO NOT REUSE THIS CODE THINKING IT VERIFIES THE CHAIN!
 */
static X509 *findEEC(STACK_OF(X509) * certstack) {
  int depth = sk_X509_num(certstack);
  if (depth == 0) {
    return nullptr;
  }
  int idx = depth - 1;
  char *priorsubject = nullptr;
  char *subject = nullptr;
  X509 *x509cert = sk_X509_value(certstack, idx);
  for (; x509cert && idx > 0; idx--) {
    subject = X509_NAME_oneline(X509_get_subject_name(x509cert), nullptr, 0);
    if (subject && priorsubject && (strncmp(subject, priorsubject, strlen(subject)) != 0)) {
      break;
    }
    x509cert = sk_X509_value(certstack, idx);
    if (subject) {
      OPENSSL_free(subject);
      subject = nullptr;
    }
  }
  if (subject) {
    OPENSSL_free(subject);
    subject = nullptr;
  }
  return x509cert;
}

static bool getX509SubjectFromFile(const std::string &filename, std::string &result) {
  BIO *biof = nullptr;
  STACK_OF(X509) *certs = nullptr;
  char *subject = nullptr;
  unsigned char *data = nullptr;
  char *header = nullptr;
  char *name = nullptr;
  long len = 0U;

  if ((biof = BIO_new_file(filename.c_str(), "r"))) {
    certs = sk_X509_new_null();
    bool encountered_error = false;
    while ((!encountered_error) && (!BIO_eof(biof)) && PEM_read_bio(biof, &name, &header, &data, &len)) {
      if (strcmp(name, PEM_STRING_X509) == 0 || strcmp(name, PEM_STRING_X509_OLD) == 0) {
        X509 *tmp_cert = nullptr;
        // See WARNINGS section in http://www.openssl.org/docs/crypto/d2i_X509.html
        // Without this cmsRun crashes on a mac with a valid grid proxy.
        const unsigned char *p;
        p = data;
        tmp_cert = d2i_X509(&tmp_cert, &p, len);
        if (tmp_cert) {
          sk_X509_push(certs, tmp_cert);
        } else {
          encountered_error = true;
        }
      }  // Note we ignore any proxy key in the file.
      if (data) {
        OPENSSL_free(data);
        data = nullptr;
      }
      if (header) {
        OPENSSL_free(header);
        header = nullptr;
      }
      if (name) {
        OPENSSL_free(name);
        name = nullptr;
      }
    }
    X509 *x509cert = nullptr;
    if (!encountered_error && sk_X509_num(certs)) {
      x509cert = findEEC(certs);
    }
    if (x509cert) {
      subject = X509_NAME_oneline(X509_get_subject_name(x509cert), nullptr, 0);
    }
    // Note we do not free x509cert directly, as it's still owned by the certs stack.
    if (certs) {
      sk_X509_pop_free(certs, X509_free);
      x509cert = nullptr;
    }
    BIO_free(biof);
    if (subject) {
      result = subject;
      OPENSSL_free(subject);
      return true;
    }
  }
  return false;
}

bool StatisticsSenderService::getX509Subject(std::string &result) {
  char *filename = getenv("X509_USER_PROXY");
  if (filename && getX509SubjectFromFile(filename, result)) {
    return true;
  }
  std::stringstream ss;
  ss << "/tmp/x509up_u" << geteuid();
  return getX509SubjectFromFile(ss.str(), result);
}
