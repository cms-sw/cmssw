# XrdAdaptor

## Introduction

The `XrdAdaptor` package is the CMSSW implementation of CMS' AAA infrastructure. The main features on top of the stock XRootD client library are
* Recovery from some errors via re-tries
* Use of multiple XRootD sources (described further [here](doc/multisource_algorithm_design.md))

## Short description of components

### `ClientRequest`

The `ClientRequest` implements `XrdCl::ResponseHandler`, and represents a single read request(?).

### `QualityMetric`

The `QualityMetric` implements a measurement of the server's recent performance for the client. It's based on the time taken to service requests (under the assumption that, since requests are split into potentially smaller-chunks by the request manager, the time to service them should be roughly the same) with a small amount of exponential weighting to prefer data points from the most recent request.

Since the metric is based on time to serve requests, a lower value is better.

Potential improvements include:
* Actually weighting the scores based on the size (or complexity) of the reads. The assumption that latency dominates transfer time may be OK in some cases -- but we've seen for large files (e.g., heavy ion), that bandwidth really is relevant and that large vector reads can cause much more server stress due to read amplification for erasure-coded systems than a simple read.
* Switching from the hand-calculated exponential weighting to a more typical exponentially weighted moving average setup.


### `RequestManager`

The `RequestManager` containes the actual implementation of the retries and the multi-source algorithm. There is one `RequestManager` object for one PFN, and it contains one or more `Source` objects.

#### `RequestManager::OpenHandler`

The `OpenHandler` implements XRootD's `XrdCl::ResponseHandler` in an asynchronous way. An instance is created in `RequestManager::initialize()`, and used when additional Sources are opened, either as part of the multi-source comparisons (`RequestManager::checkSourcesImpl()`) or read error recovery (`RequestManager::requestFailure()`).

Most of the internal operations for the xrootd client are asynchronous while ROOT expects a synchronous interface from the adaptor. The difference between the two here is the asynchronous one is used for "background probing" to find additional sources.

### `Source`

The `Source` represents a connection to one storage server. There can be more than one `Source` for one PFN. 

### `SyncHostResponseHandler`

The `SyncHostResponseHandler` implements XRootD's `XrdCl::ResponseHandler` in a synchronous way(?). It is used in `RequestManager::initialize()` for the initial file open.

### `XrdFile`

The `XrdFile` implements `edm::storage::Storage` (see [Utilities/StorageFactory/README.md](../../Utilities/StorageFactory/README.md). In CMS' terminology it represents one Physical File Name (PFN), and acts as a glue between the `edm::storage::Storage` API and `RequestManager`.

### `XrdStatistics`

The `XrdStatistics` provides per-"site" counters (bytes read, total time), providing the `XrdStatisticsService` with a viewpoint of how individual sites are performing for a given client. The intent is to provide more visibility into the performance encountered by grouping I/O operations into a per-site basis, under the theory that performance within a site is similar but could differ between two different sites.

### `XrdStatisticsService`

The `XrdStatisticsService` is a Service to report XrootD-related statistics centrally. It is one of the default Services that are enabled in `cmsRun`.

### `XrdStorageMaker`

The `XrdStorageMaker` is a plugin in the `StorageMaker` hierarchy. See [Utilities/StorageFactory/README.md](../../Utilities/StorageFactory/README.md) for more information. Among other things it creates `XrdFile` objects.
