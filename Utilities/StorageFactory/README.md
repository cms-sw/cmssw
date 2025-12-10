# Utilities/StorageFactory Description

## Introduction
The package provides a mechanism to customize communication with a storage system in a file format agnostic manner. The storage system is designated via its protocol name, e.g. `file:`, `http:`, `root:`.

## `edm::storage::Storage`
This is the abstract interface class for handling read/write operations to the underlying storage system. 

## `edm::storage::StorageFactory`
Factory interface for constructing `edm::storage::Storage` instances. Also provides setting/getting default storage system values to be used for the job (e.g. doing account summary).
`StorageFactory` provides two implementations of `edm::storage::Storage` classes which can be used to wrap around any other `Storage` object.

###  `edm::storage::LocalCacheFile`
Does memory mapped caching of the wrapped `Storage` object.  This is only applied if `CACHE_HINT_LAZY_DOWNLOAD` is set for `cacheHint` or the protocol handling code explicit passes `IOFlags::OpenWrap` to `StorageFactory::wrapNonLocalFile`. The wrapping does not happen if the Storage is open for writing nor if the Storage is associated with a file on the local file system. Note that files using the `file:` protocol _can_ end up using `LocalCacheFile` if the path is determined to be on a non-local file system.

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


## Generic storage proxies

This facility resembles the `edm::storage::LocalCacheFile` and `edm::storage::StorageAccountProxy` in the way that `edm::storage::Storage` objects constructed by the concrete `edm::storage::StorageMaker` are wrapped into other `edm::storage::Storage` objects.

The proxies are configured via `TFileAdaptor`'s `storageProxies` `VPSet` configuration parameter. The proxies are wrapped in the order they are specified in the `VPSet`, i.e. the first element wraps the concrete `edm::storage::Storage`, second element wraps the first element etc. The `edm::storage::StorageAccountProxy` and `edm::storage::LocalCacheFile` wrap the last storage proxy according to their usual behavior.

Each concrete proxy comes with two classes, the proxy class itself (inheriting from the `edm::storage::StorageProxyBase`) and a maker class (inheriting from the `edm::storage::StorageProxyMaker`). This "factory of factories" pattern is used because a maker is created once per job (in `TFileAdaptor`), and the maker object is used to create a proxy object for each file.

### Concrete proxy classes

The convention is to use the proxy class name as the plugin name for the maker, as the proxy is really what the user would care for. The headings of the subsections correspond to the plugin names.

#### `StorageTracerProxy`

The `edm::storage::StorageTracerProxy` (and the corresponding `edm::storage::StorageTracerProxyMaker`) produces a text file with a trace of all IO operations at the `StorageFactory` level. The behavior of each concrete `Storage` object (such as further splitting of read requests in `XrdAdaptor`) is not captured in these tracers. The structure of the trace file is described in a preamble in the trace file.

The plugin has a configuration parameter for a pattern for the trace files. The pattern must contain at least one `%I`. The maker has an atomic counter for the files, and all occurrences of `%I` are replaced with the value of that counter for the given file.

There is an `edmStorageTracer.py` script for doing some analyses of the traces.

The `StorageTracerProxy` also provides a way to correlate the trace entries with the rest of the framework via [MessageLogger](../../FWCore/MessageService/Readme.md) messages. These messages are issued with the DEBUG severity and `IOTrace` category. There are additional, higher-level messages as part of the `PoolSource`. To see these messages, compile the `Utilities/Storage` and `IOPool/Input` packages with `USER_CXXFLAGS="-DEDM_ML_DEBUG", and customize the MessageLogger configuration along
```py
process.MessageLogger.cerr.threshold = "DEBUG"
```

#### `StorageAddLatencyProxy`

The `edm::storage::StorageAddLatencyProxy` (and the corresponding `edm::storage::StorageAddLatencyProxyMaker`) can be used to add artifical latency to the IO operations. The plugin has configuration parameters for latencies of singular reads, vector reads, singular writes, and vector writes.

If used together with `StorageTracerProxy` to e.g. simulate the behavior of high-latency storage systems with e.g. local files, the `storageProxies` `VPSet` should have `StorageAddLatencyProxy` first, followed by `StorageTracerProxy`.

### Other components

#### `edm::storage::StorageProxyBase`

Inherits from `edm::storage::Storage` and is the base class for the proxy classes.

#### `edm::storage::StorageProxyMaker`

Base class for the proxy makers.


## Related classes in other packages

### TStorageFactoryFile
Inherits from `TFile` but uses `edm::storage::Storage` instances when doing the actual read/write operations. The class explicitly uses `"tstoragefile"` when communicating with `edm::storage::StorageAccount`.

### `TFileAdaptor`

`TFileAdaptor` is a cmsRun Service (with a plugin name of `AdaptorConfig`, see [FWStorage/TFileAdaptor/README.md](../../FWStorage/TFileAdaptor/README.md)). It explicitly registers the use of `TStorageFactoryFile` with ROOT's `TFile::Open` system. The parameters passed to `TFileAdaptor` are relayed to `edm::storage::StorageFactory` to setup the defaults for the job.

### `CondorStatusService`
Sends condor _Chirp_ messages periodically from cmsRun. These include the most recent aggregated `edm::storage::StorageAccount` information for all protocols being used except for the `"tstoragefile"` protocol.

### `StatisticsSenderService`
A cmsRun Service which sends out UDP packets about the state of the system. The information is sent when a primary file closes and includes the recent aggregated `edm::storage::StorageAccount` information for all protocols being used except for the `"tstoragefile"` protocol.

### `XrdAdaptor`

A `edm::storage::Storage` implementation for xrootd (see [Utilities/XrdAdaptor/README.md](../../Utilities/XrdAdaptor/README.md)).
