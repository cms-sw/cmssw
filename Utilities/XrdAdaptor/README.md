# XrdAdaptor

## Introduction

The `XrdAdaptor` package is the CMSSW implementation of CMS' AAA infrastructure. The main features on top of the stock XRootD client library are
* Recovery from some errors via re-tries
* Use of multiple XRootD sources (described further [here](doc/multisource_algorithm_design.md))

## Short description of components

### `ClientRequest`

The `ClientRequest` implements `XrdCl::ResponseHandler`, and represents a single read request(?).

### `QualityMetric` etc

?

### `RequestManager`

The `RequestManager` containes the actual implementation of the retries and the multi-source algorithm. There is one `RequestManager` object for one PFN, and it contains one or more `Source` objects.

#### `RequestManager::OpenHandler`

The `OpenHandler` implements XRootD's `XrdCl::ResponseHandler` in an asynchronous way(?). An instance is created in `RequestManager::initialize()`, and used when additional Sources are opened, either as part of the multi-source comparisons (`RequestManager::checkSourcesImpl()`) or read error recovery (`RequestManager::requestFailure()`).

### `Source`

The `Source` represents a connection to one storage server. There can be more than one `Source` for one PFN. 

### `SyncHostResponseHandler`

The `SyncHostResponseHandler` implements XRootD's `XrdCl::ResponseHandler` in a synchronous way(?). It is used in `RequestManager::initialize()` for the initial file open.

### `XrdFile`

The `XrdFile` implements `edm::storage::Storage` (see [Utilities/StorageFactory/README.md](../../Utilities/StorageFactory/README.md). In CMS' terminology it represents one Physical File Name (PFN), and acts as a glue between the `edm::storage::Storage` API and `RequestManager`.

### `XrdStatistics`

?

### `XrdStatisticsService`

The `XrdStatisticsService` is a Service to report XrootD-related statistics centrally. It is one of the default Services that are enabled in `cmsRun`.

### `XrdStorageMaker`

The `XrdStorageMaker` is a plugin in the `StorageMaker` hierarchy. See [Utilities/StorageFactory/README.md](../../Utilities/StorageFactory/README.md) for more information. Among other things it creates `XrdFile` objects.
