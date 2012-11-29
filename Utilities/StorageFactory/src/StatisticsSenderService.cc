
#include "Utilities/StorageFactory/interface/StatisticsSenderService.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/src/Guid.h"

#include <string>

#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#include <openssl/x509.h>
#include <openssl/pem.h>

#define UPDATE_STATISTIC(x) \
    m_ ## x = x;

#define UPDATE_AND_OUTPUT_STATISTIC(x) \
    os << #x "=" << (x-m_ ## x) << ", "; \
    UPDATE_STATISTIC(x)

// Simple hack to define HOST_NAME_MAX on Mac.
// Allows arrays to be statically allocated
#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 128
#endif

#define JOB_UNIQUE_ID_ENV "CRAB_UNIQUE_JOB_ID"

using namespace edm::storage;

StatisticsSenderService::FileStatistics::FileStatistics() :
  m_read_single_operations(0),
  m_read_single_bytes(0),
  m_read_single_square(0),
  m_read_vector_operations(0),
  m_read_vector_bytes(0),
  m_read_vector_square(0),
  m_read_vector_count_sum(0),
  m_read_vector_count_square(0),
  m_read_bytes_at_close(0),
  m_start_time(time(NULL))
{}

void
StatisticsSenderService::FileStatistics::fillUDP(std::ostringstream &os) {
  const StorageAccount::StorageStats &stats = StorageAccount::summary();
  ssize_t read_single_operations = 0;
  ssize_t read_single_bytes = 0;
  ssize_t read_single_square = 0;
  ssize_t read_vector_operations = 0;
  ssize_t read_vector_bytes = 0;
  ssize_t read_vector_square = 0;
  ssize_t read_vector_count_sum = 0;
  ssize_t read_vector_count_square = 0;
  for (StorageAccount::StorageStats::const_iterator i = stats.begin (); i != stats.end(); ++i) {
    if (i->first == "tstoragefile") {
      continue;
    }
    for (StorageAccount::OperationStats::const_iterator j = i->second->begin(); j != i->second->end(); ++j) {
      if (j->first == "readv") {
        read_vector_operations += j->second.attempts;
        read_vector_bytes += j->second.amount;
        read_vector_count_square += j->second.vector_square;
        read_vector_square += j->second.amount_square;
        read_vector_count_sum += j->second.vector_count;
      } else if (j->first == "read") {
        read_single_operations += j->second.attempts;
        read_single_bytes += j->second.amount;
        read_single_square += j->second.amount_square;
      }
    }
  }
  int64_t single_op_count = read_single_operations - m_read_single_operations;
  if (single_op_count > 0) {
    double single_sum = read_single_bytes-m_read_single_bytes;
    double single_average = single_sum/static_cast<double>(single_op_count);
    os << "read_single_sigma:" << sqrt((static_cast<double>(read_single_square-m_read_single_square) - single_average*single_average*single_op_count)/static_cast<double>(single_op_count)) << ", ";
    os << "read_single_average:" << single_average << ", ";
  }
  m_read_single_square = read_single_square;
  int64_t vector_op_count = read_vector_operations - m_read_vector_operations;
  if (vector_op_count > 0) {
    double vector_average = static_cast<double>(read_vector_bytes-m_read_vector_bytes)/static_cast<double>(vector_op_count);
    os << "read_vector_average:" << vector_average << ", ";
    os << "read_vector_sigma:" << sqrt((static_cast<double>(read_vector_square-m_read_vector_square) - vector_average*vector_average*vector_op_count)/static_cast<double>(vector_op_count)) << ", ";
    double vector_count_average = static_cast<double>(read_vector_count_sum-m_read_vector_count_sum)/static_cast<double>(vector_op_count);
    os << "read_vector_count_average:" << vector_count_average << ", ";
    os << "read_vector_count_sigma:" << sqrt((static_cast<double>(read_vector_count_square-m_read_vector_count_square) - vector_count_average*vector_count_average*vector_op_count)/static_cast<double>(vector_op_count)) << ", ";
  }
  m_read_vector_square = read_vector_square;
  m_read_vector_count_square = read_vector_count_square;
  m_read_vector_count_sum = read_vector_count_sum;
  // See top of file for macros; not complex, just avoiding copy/paste
  UPDATE_AND_OUTPUT_STATISTIC(read_single_operations)
  UPDATE_AND_OUTPUT_STATISTIC(read_single_bytes)
  UPDATE_AND_OUTPUT_STATISTIC(read_vector_operations)
  UPDATE_AND_OUTPUT_STATISTIC(read_vector_bytes)

  os << "read_bytes:" << (m_read_vector_bytes+m_read_single_bytes) << ", ";
  os << "read_bytes_at_close:" << (m_read_vector_bytes+m_read_single_bytes) << ", ";

  os << "start_time:" << m_start_time << ", ";
  m_start_time = time(NULL);
  // NOTE: last entry doesn't have the trailing comma.
  os << "end_time:" << m_start_time;
}

StatisticsSenderService::StatisticsSenderService(edm::ParameterSet const& /*pset*/, edm::ActivityRegistry& ar) :
  m_clienthost("unknown"),
  m_clientdomain("unknown"),
  m_serverhost("unknown"),
  m_serverdomain("unknown"),
  m_filelfn("unknown"),
  m_filestats(),
  m_guid(Guid().toString()),
  m_counter(0),
  m_size(-1),
  m_userdn("unknown")
{
  determineHostnames();
  ar.watchPreCloseFile(this, &StatisticsSenderService::filePreCloseEvent);
  if (!getX509Subject(m_userdn)) {
    m_userdn = "unknown";
  }
}

const char *
StatisticsSenderService::getJobID() {
  return getenv(JOB_UNIQUE_ID_ENV);
}

void
StatisticsSenderService::setCurrentServer(const std::string &servername) {
  size_t dot_pos = servername.find(".");
  if (dot_pos == std::string::npos) {
    m_serverhost = servername.substr(0, servername.find(":"));
    m_serverdomain = "unknown";
  } else {
    m_serverhost = servername.substr(0, dot_pos);
    m_serverdomain = servername.substr(dot_pos+1, servername.find(":")-dot_pos-1);
    if (m_serverdomain.empty()) {
      m_serverdomain = "unknown";
    }
  }
}

void
StatisticsSenderService::setSize(size_t size) {
  m_size = size;
}

void
StatisticsSenderService::filePreCloseEvent(std::string const& lfn, bool usedFallback) {
  m_filelfn = lfn;

  edm::Service<edm::SiteLocalConfig> pSLC;
  if (!pSLC.isAvailable()) {
    return;
  }

  const struct addrinfo * addresses = pSLC->statisticsDestination();
  if (!addresses) {
    return;
  }

  std::string results;
  fillUDP(pSLC->siteName(), usedFallback, results);

  for (const struct addrinfo *address = addresses; address != NULL; address = address->ai_next) {
    int sock = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
    if (sock < 0) {
      continue;
    }
    if (sendto(sock, results.c_str(), results.size(), 0, address->ai_addr, address->ai_addrlen) >= 0) {
      break; 
    }
  }

  m_counter++;
}

void
StatisticsSenderService::determineHostnames(void) {
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
    m_clientdomain = m_clienthost.substr(dot_pos+1, m_clienthost.size()-dot_pos-1);
    m_clienthost = m_clienthost.substr(0, dot_pos);
  }
}

void
StatisticsSenderService::fillUDP(const std::string& siteName, bool usedFallback, std::string &udpinfo) {
  std::ostringstream os;

  // Header - same for all IO accesses
  os << "{";
  if (!siteName.empty()) {
    os << "site_name:\"" << siteName << "\", ";
  }
  if (usedFallback) {
    os << "fallback: true, ";
  }
  os << "user_dn:\"" << m_userdn << "\", ";
  os << "client_host:\"" << m_clienthost << "\", ";
  os << "client_domain:\"" << m_clientdomain << "\", ";
  os << "server_host:\"" << m_serverhost << "\", ";
  os << "server_domain:\"" << m_serverdomain << "\", ";
  os << "unique_id:\"" << m_guid << "-" << m_counter << "\", ";
  os << "file_lfn:\"" << m_filelfn << "\", ";
  const char * jobId = getJobID();
  if (jobId) {
    os << "app_info:\"" << jobId << "\", ";
  } else {
    os << "app_info:\"" << m_guid << "\", ";
  }

  if (m_size >= 0) {
    os << "file_size:" << m_size << ", ";
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
static X509 * findEEC(STACK_OF(X509) * certstack) {
  int depth = sk_X509_num(certstack);
  if (depth == 0) {
    return NULL;
  }
  int idx = depth-1;
  char *priorsubject = NULL, *subject;
  X509 *x509cert = sk_X509_value(certstack, idx);
  for (; x509cert && idx>0; idx--) {
    subject = X509_NAME_oneline(X509_get_subject_name(x509cert),0,0);
    if (subject && priorsubject && (strncmp(subject, priorsubject, strlen(subject)) != 0)) {
      break;
    }
    x509cert = sk_X509_value(certstack, idx);
    if (subject) {
      OPENSSL_free(subject);
      subject = NULL;
    }
  }
  if (subject) {
    OPENSSL_free(subject);
    subject = NULL;
  }
  return x509cert;
}

static bool
getX509SubjectFromFile(const std::string &filename, std::string &result) {
  BIO *biof = NULL;
  STACK_OF(X509) *certs = NULL;
  char *subject=NULL;
  unsigned char *data;
  char *header = NULL;
  char *name = NULL;
  long len;

  if((biof = BIO_new_file(filename.c_str(), "r")))  {

    certs = sk_X509_new_null();
    bool encountered_error = false;
    while ((!encountered_error) && (!BIO_eof(biof)) && PEM_read_bio(biof, &name, &header, &data, &len)) {
      if (strcmp(name, PEM_STRING_X509) == 0 || strcmp(name, PEM_STRING_X509_OLD) == 0) {
        X509 * tmp_cert = NULL;
        tmp_cert = d2i_X509(&tmp_cert, const_cast<const unsigned char **>(&data), len);
        if (tmp_cert) {
          sk_X509_push(certs, tmp_cert);
        } else {
          encountered_error = true;
        }
      } // Note we ignore any proxy key in the file.
      if (data) { OPENSSL_free(data); data = NULL;}
      if (header) { OPENSSL_free(header); header = NULL;}
      if (name) { OPENSSL_free(name); name = NULL;}
    }
    X509 *x509cert = NULL;
    if (!encountered_error && sk_X509_num(certs)) {
      x509cert = findEEC(certs);
    }
    if (x509cert) {
      subject = X509_NAME_oneline(X509_get_subject_name(x509cert),0,0);
    }
    // Note we do not free x509cert directly, as it's still owned by the certs stack.
    if (certs) {
      sk_X509_pop_free(certs, X509_free);
      x509cert = NULL;
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

bool
StatisticsSenderService::getX509Subject(std::string &result) {
  char *filename = getenv("X509_USER_PROXY");
  if (filename && getX509SubjectFromFile(filename, result)) {
    return true;
  }
  std::stringstream ss;
  ss << "/tmp/x509up_u" << geteuid();
  return getX509SubjectFromFile(ss.str(), result);
}
