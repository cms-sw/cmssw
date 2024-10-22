# Utilities/StorageFactory Description

## Introduction
The package provides a mechanism to customize communication with a storage system in a file format agnostic manner. The storage system is designated via its protocol name, e.g. `file:`, `http:`, `root:`.

## `edm::storage::Storage`
This is the abstract interface class for handling read/write operations to the underlying storage system. 

## `edm::storage::StorageFactory`
Factory interface for constructing `edm::storage::Storage` instances. Also provides setting/getting default storage system values to be used for the job (e.g. doing account summary).
`StorageFactory` provides two implementations of `edm::storage::Storage` classes which can be used to wrap around any other `Storage` object.

###  `edm::storage::LocalCacheFile`
Does memory mapped caching of the wrapped `Storage` object.  This is only applied if `CACHE_HINT_LAZY_DOWNLOAD` is set for `cacheHint` or the protocol handling code explicit passes `IOFlags::OpenWrap` to `StorageFactory::wrapNonLocalFile`. The wrapping does not happen if the Storage is open for writing nor if the Storage is associated with a file on the local file system.

### `edm::storage::StorageAccountProxy`
This wraps the `Storage` object and provides per protocol accounting information (e.g. number of bytes read) to `edm::storage::StorageAccount`. This is only used if `StorageFactory::accounting()` returns `true`.

## `edm::storage::StorageMakerFactory`
Used by `edm::storage::StorageFactory` to dynamically load factory methods for the given `edm::storage::Storage` implementations.

## `edm::storage::StorageMaker`
Base class for factory classes that construct concrete versions of `edm::storage::Storage` objects. One creates and registers one of these classes for each storage protocol.

## `edm::storage::StorageAccount`
A singleton used to aggragate statistics about all storage calls for each protocol being used by a job.
### `edm::storage::StorageAccount::StorageClassToken`
Each protocol is associated to a token for quick lookup.

## Related classes in other packages

### TStorageFactoryFile
Inherits from `TFile` but uses `edm::storage::Storage` instances when doing the actual read/write operations. The class explicitly uses `"tstoragefile"` when communicating with `edm::storage::StorageAccount`.

### TFileAdaptor
TFileAdaptor is a cmsRun Service. It explicitly registers the use of `TStorageFactoryFile` with ROOT's `TFile::Open` system. The parameters passed to `TFileAdaptor` are relayed to `edm::storage::StorageFactory` to setup the defaults for the job.

### CondorStatusService
Sends condor _Chirp_ messages periodically from cmsRun. These include the most recent aggregated `edm::storage::StorageAccount` information for all protocols being used except for the `"tstoragefile"` protocol.

### StatisticsSenderService
A cmsRun Service which sends out UDP packets about the state of the system. The information is sent when a primary file closes and includes the recent aggregated `edm::storage::StorageAccount` information for all protocols being used except for the `"tstoragefile"` protocol.
