
/*
 * These functions are a re-implementation of upstream's readv.
 * The important aspect is we have vectored scatter-gather IO.
 * In the upstream readv, the vectored IO goes into one buffer - 
 * not scatter gathered.
 *
 * CMSSW now uses scatter-gather in the TFileAdapter's ReadReapacker.
 * Hence, we have to emulate it using XrdClient::ReadV - horribly slow!
 *
 * Why not continue to use the XrdClient's internal cache?  Each time we use a
 * different TTC, it invalidates the cache.  So, the internal cache and our
 * trigger-pattern TTC usage forces a use of readv instead of prefetch.
 *
 */

#include "Utilities/XrdAdaptor/src/XrdFile.h"
#include "XProtocol/XProtocol.hh"
#include "XrdClient/XrdClientProtocol.hh"
#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientSid.hh"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Likely.h"

#include <assert.h>

class MutexSentry
{
public:
  MutexSentry(pthread_mutex_t &mutex) : m_mutex(mutex) {pthread_mutex_lock(&m_mutex);}

 ~MutexSentry() {pthread_mutex_unlock(&m_mutex);}

private:
 pthread_mutex_t &m_mutex;

};

// This method is rarely used by CMS; hence, it is a small wrapper and not efficient.
IOSize
XrdFile::readv (IOBuffer *into, IOSize n)
{
  vector<IOPosBuffer> new_buf;
  new_buf.reserve(n);
  IOOffset off = 0;
  for (IOSize i=0; i<n; i++) {
    IOSize size = into[i].size();
    new_buf[i] = IOPosBuffer(off, into[i].data(), size);
    off += size;
  }
  return readv(&(new_buf[0]), n);
}

/*
 * A vectored scatter-gather read.
 * Returns the total number of bytes successfully read.
 *
 */
IOSize
XrdFile::readv (IOPosBuffer *into, IOSize n)
{
  assert(m_client);
  
  // A trivial vector read - unlikely, considering ROOT data format.
  if (unlikely(n == 0)) {
    return 0;
  }
  if (unlikely(n == 1)) {
    return read(into[0].data(), into[0].size(), into[0].offset());
  }

  // The main challenge here is to turn the request into a form that can be
  // fed to the Xrootd connection.  In particular, Xrootd has a limit on the
  // maximum number of chunks and the maximum size of each chunk.  Hence, the
  // loop unrolling.
  // 
  IOSize total_len = 0;
  readahead_list read_chunk_list[READV_MAXCHUNKS];
  char *result_list[READV_MAXCHUNKS];
  IOSize chunk_off = 0;
  IOSize readv_total_len = 0;
  const char * handle = m_client->GetHandle(); // also - 16 bytes offset from the location of m_client.
  for (IOSize i = 0; i < n; ++i) {

    IOSize len = into[i].size();
    if (unlikely(len > 0x7fffffff)) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdFile::readv(name='" << m_name << "')[" << i
         << "].size=" << len << " exceeds read size limit 0x7fffffff";
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
      throw ex;
    }

    IOOffset off = into[i].offset();
    char *chunk_data = static_cast<char *>(into[i].data());
    while (len > 0) { // Iterate as long as there is additional data to read.
                      // Each iteration will read up to READV_MAXCHUNKSIZE of this request.
      IOSize chunk_size = len > READV_MAXCHUNKSIZE ? READV_MAXCHUNKSIZE : len;
      len -= chunk_size;
      readv_total_len += chunk_size;
      read_chunk_list[chunk_off].rlen = chunk_size;
      read_chunk_list[chunk_off].offset = off;
      result_list[chunk_off] = chunk_data;
      chunk_data += chunk_size;
      off += chunk_size;
      memcpy(&(read_chunk_list[chunk_off].fhandle), handle, 4);
      chunk_off++;
      if (chunk_off == READV_MAXCHUNKS) {
        // Now that we have broken the readv into Xrootd-sized chunks, send the actual command.
        // readv_send will also parse the response and place the data into the result_list buffers.
        IOSize tmp_total_len = readv_send(result_list, *read_chunk_list, chunk_off, readv_total_len);
        total_len += tmp_total_len;
        if (tmp_total_len != readv_total_len)
          return total_len;
        readv_total_len = 0;
        chunk_off = 0;
      }
    }
  }
  // Do the actual readv for all remaining chunks.
  if (chunk_off) {
    total_len += readv_send(result_list, *read_chunk_list, chunk_off, readv_total_len);
  }
  return total_len;
}

/*
 * Send the readv request to Xrootd.
 * Returns the number of bytes stored into result_list.
 * Assumes that read_chunk_list and result_list are of size n; the results of the reads
 * described in read_chunk_list will be stored in the buffer pointed to by result_list.
 * total_len should be the sum of the size of all reads.
 */
IOSize
XrdFile::readv_send(char **result_list, readahead_list &read_chunk_list, IOSize n, IOSize total_len)
{
  // Per the xrootd protocol document:
  // Sending requests using the same streamid when a kXR_oksofar status code has been 
  // returned may produced unpredictable results. A client must serialize all requests 
  // using the streamid in the presence of partial results.

  XrdClientConn *xrdc = m_client->GetClientConn();
  ClientRequest readvFileRequest;
  memset( &readvFileRequest, 0, sizeof(readvFileRequest) );
  
  kXR_unt16 sid = ConnectionManager->SidManager()->GetNewSid();
  memcpy(readvFileRequest.header.streamid, &sid, sizeof(kXR_unt16));
  readvFileRequest.header.requestid = kXR_readv;
  readvFileRequest.readv.dlen = n * sizeof(struct readahead_list);

  std::vector<char> res_buf;
  res_buf.reserve( total_len + (n * sizeof(struct readahead_list)) );

  // Encode, then send the command.
  clientMarshallReadAheadList(&read_chunk_list, readvFileRequest.readv.dlen);
  bool success;
  IOSize data_length;
  {
    MutexSentry sentry(m_readv_mutex);
    success = xrdc->SendGenCommand(&readvFileRequest, &read_chunk_list, 0,
                                   (void *)&(res_buf[0]), FALSE, (char *)"ReadV");
    data_length = xrdc->LastServerResp.dlen;
  }
  clientUnMarshallReadAheadList(&read_chunk_list, readvFileRequest.readv.dlen);

  ConnectionManager->SidManager()->ReleaseSid(sid);

  if (success) {
    return readv_unpack(result_list, res_buf, data_length, read_chunk_list, n);
  } else {
    return 0;
  }

}

/*
 * Unpack the response buffer from Xrootd into the final results buffer.
 */
IOSize
XrdFile::readv_unpack(char **result_list, std::vector<char> &result_buf, IOSize response_length, readahead_list &read_chunk_list, IOSize n)
{
  IOSize response_offset = 0;
  IOSize total_len = 0;
  for (IOSize i = 0; i < n; i++) {

    if (unlikely(response_offset + sizeof(struct readahead_list) > response_length)) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdFile::readv(name='" << m_name << "')[" << i
         << "] returned an incorrectly-sized response (short header)";
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
    }

    kXR_int64 offset;
    kXR_int32 rlen;
    { // Done as a separate block so response is not used later - as it is all in network order!
      const readahead_list *response = reinterpret_cast<struct readahead_list*>(&result_buf[response_offset]);
      offset = ntohll(response->offset);
      rlen = ntohl(response->rlen);
    }

    // Sanity / consistency checks; verify the results correspond to the requested chunk
    // Also check that the response buffer is sufficient large to read from.
    if (unlikely((&read_chunk_list)[i].offset != offset)) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdFile::readv(name='" << m_name << "')[" << i
         << "] returned offset " << offset << " does not match requested offset "
         << (&read_chunk_list)[i].offset; 
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
      throw ex;
    }
    if (unlikely((&read_chunk_list)[i].rlen != rlen)) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdFile::readv(name='" << m_name << "')[" << i
         << "] returned size " << rlen << " does not match requested size "
         << (&read_chunk_list)[i].rlen;
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
      throw ex;
    }
    if (unlikely(response_offset + rlen > response_length)) {
      edm::Exception ex(edm::errors::FileReadError);
      ex << "XrdFile::readv(name='" << m_name << "')[" << i
         << "] returned an incorrectly-sized response (short data)";
      ex.addContext("Calling XrdFile::readv()");
      addConnection(ex);
    }

    response_offset += sizeof(struct readahead_list); // Data is stored after header.
    total_len += rlen;
    // Copy the data into place; increase the offset.
    memcpy(result_list[i], &result_buf[response_offset], rlen);
    response_offset += rlen;
  }

  return total_len;
}

