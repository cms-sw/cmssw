
#ifndef Utilities_StorageFactory_StatisticsSenderService_H
#define Utilities_StorageFactory_StatisticsSenderService_H

#include <string>
#include <sstream>

namespace edm {

  class ParameterSet; 
  class ActivityRegistry;

  namespace storage {

    class StatisticsSenderService {
      public:
        StatisticsSenderService(edm::ParameterSet const& pset, edm::ActivityRegistry& ar);

        void setSize(size_t size);
        void setCurrentServer(const std::string &servername);
        void filePreCloseEvent(std::string const& lfn, bool usedFallback);
        static const char * getJobID();
        static bool getX509Subject(std::string &);
      private:

        class FileStatistics {
          public:
            FileStatistics();
            void fillUDP(std::ostringstream &os);
          private:
            ssize_t m_read_single_operations;
            ssize_t m_read_single_bytes;
            ssize_t m_read_single_square;
            ssize_t m_read_vector_operations;
            ssize_t m_read_vector_bytes;
            ssize_t m_read_vector_square;
            ssize_t m_read_vector_count_sum;
            ssize_t m_read_vector_count_square;
            time_t  m_start_time;
        };

        void determineHostnames(void);
        void fillUDP(const std::string&, bool, std::string &);
        std::string    m_clienthost;
        std::string    m_clientdomain;
        std::string    m_serverhost;
        std::string    m_serverdomain;
        std::string    m_filelfn;
        FileStatistics m_filestats;
        std::string    m_guid;
        size_t         m_counter;
        ssize_t        m_size;
        std::string    m_userdn;
    };

    
  }
}

#endif

