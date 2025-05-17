#include "FWCore/Concurrency/interface/ThreadSafeOutputFileStream.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/StorageProxyMaker.h"

#include <atomic>
#include <chrono>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <boost/algorithm/string.hpp>
#include <fmt/format.h>

namespace edm::storage {
  class StorageTracerProxy : public Storage {
    static constexpr std::string_view kOpen = "o";
    static constexpr std::string_view kRead = "r";
    static constexpr std::string_view kReadv = "rv";
    static constexpr std::string_view kReadvElement = "rve";
    static constexpr std::string_view kWrite = "w";
    static constexpr std::string_view kWritev = "wv";
    static constexpr std::string_view kWritevElement = "wve";
    static constexpr std::string_view kPosition = "s";
    static constexpr std::string_view kPrefetch = "p";
    static constexpr std::string_view kPrefetchElement = "pe";
    static constexpr std::string_view kResize = "rsz";
    static constexpr std::string_view kFlush = "f";
    static constexpr std::string_view kClose = "c";

  public:
    StorageTracerProxy(unsigned id,
                       std::string const& tracefile,
                       std::string const& storageUrl,
                       std::unique_ptr<Storage> storage)
        : file_(tracefile), baseStorage_(std::move(storage)), traceId_(id) {
      using namespace std::literals::string_literals;
      file_.write(
          "# Format\n"s + "# --------\n"s + "# prefixes\n"s + "# #: comment\n"s +
          fmt::format("# {}: file open\n", kOpen) + fmt::format("# {}: singular read\n", kRead) +
          fmt::format("# {}: vector read\n", kReadv) +
          fmt::format("# {}: vector read element of the preceding '{}' line\n", kReadvElement, kReadv) +
          fmt::format("# {}: singular write\n", kWrite) + fmt::format("# {}: vector write\n", kWritev) +
          fmt::format("# {}: vector write element of the preceding '{}' line\n", kWritevElement, kWritev) +
          fmt::format("# {}: position (seek)\n", kPosition) + fmt::format("# {}: prefetch\n", kPrefetch) +
          fmt::format("# {}: prefetch element of the preceding '{}' line\n", kPrefetch, kPrefetchElement) +
          fmt::format("# {}: resize\n", kResize) + fmt::format("# {}: flush\n", kFlush) +
          fmt::format("# {}: close\n", kClose) + "# --------\n"s + "# line formats\n"s +
          fmt::format("# {} <id> <timestamp ms> <file name>\n", kOpen) +
          fmt::format("# {} <id> <timestamp ms> <duration us> <offset B> <requested B> <actual B>\n", kRead) +
          fmt::format(
              "# {} <id> <timestamp ms> <duration us> <requested total B> <actual total B> <number of elements>\n",
              kReadv) +
          fmt::format("# {} <index> <offset B> <requested B>\n", kReadvElement) +
          fmt::format("# {} <id> <timestamp ms> <duration us> <offset B> <requested B> <actual B>\n", kWrite) +
          fmt::format(
              "# {} <id> <timestamp ms> <duration us> <requested total B> <actual total B> <number of elements>\n",
              kWritev) +
          fmt::format("# {} <index> <offset B> <requested B>\n", kWritevElement) +
          fmt::format("# {} <id> <timestamp ms> <duration us> <offset B> <whence>\n", kPosition) +
          fmt::format("# {} <id> <timestamp ms> <duration us> <requested total B> <number of elements> <supported?>\n",
                      kPrefetch) +
          fmt::format("# {} <index> <offset B> <requested B>\n", kPrefetchElement) +
          fmt::format("# {} <id> <timestamp ms> <duration us> <size B>\n", kResize) +
          fmt::format("# {} <id> <timestamp ms> <duration us>\n", kFlush) +
          fmt::format("# {} <id> <timestamp ms> <duration us>\n", kClose) + "# --------\n"s);
      auto const entryId = idCounter_.fetch_add(1);
      file_.write(fmt::format("{} {} {} {}\n",
                              kOpen,
                              entryId,
                              std::chrono::round<std::chrono::milliseconds>(now().time_since_epoch()).count(),
                              storageUrl));
      LogTrace("IOTrace").format("IOTrace {} id {}", traceId_, entryId);
    }

    IOSize read(void* into, IOSize n) override {
      auto const offset = baseStorage_->position();
      auto const [result, message] = operate([this, into, n]() { return baseStorage_->read(into, n); });
      file_.write(fmt::format("{} {} {} {} {}\n", kRead, message, offset, n, result));
      return result;
    }

    IOSize read(void* into, IOSize n, IOOffset pos) override {
      auto const [result, message] = operate([this, into, n, pos]() { return baseStorage_->read(into, n, pos); });
      file_.write(fmt::format("{} {} {} {} {}\n", kRead, message, pos, n, result));
      return result;
    }

    IOSize readv(IOBuffer* into, IOSize n) override {
      auto offset = baseStorage_->position();
      auto const [result, message] = operate([this, into, n]() { return baseStorage_->readv(into, n); });
      std::string elements;
      IOSize total = 0;
      for (IOSize i = 0; i < n; ++i) {
        elements += fmt::format("{} {} {} {}\n", kReadvElement, i, offset, into[i].size());
        total += into[i].size();
        offset += into[i].size();
      }
      file_.write(fmt::format("{} {} {} {} {}\n", kReadv, message, total, result, n) + elements);
      return result;
    }

    IOSize readv(IOPosBuffer* into, IOSize n) override {
      auto const [result, message] = operate([this, into, n]() { return baseStorage_->readv(into, n); });
      std::string elements;
      IOSize total = 0;
      for (IOSize i = 0; i < n; ++i) {
        elements += fmt::format("{} {} {} {}\n", kReadvElement, i, into[i].offset(), into[i].size());
        total += into[i].size();
      }
      file_.write(fmt::format("{} {} {} {} {}\n", kReadv, message, total, result, n) + elements);
      return result;
    }

    IOSize write(const void* from, IOSize n) override {
      auto const offset = baseStorage_->position();
      auto const [result, message] = operate([this, from, n]() { return baseStorage_->write(from, n); });
      file_.write(fmt::format("{} {} {} {} {}\n", kWrite, message, offset, n, result));
      return result;
    }

    IOSize write(const void* from, IOSize n, IOOffset pos) override {
      auto const [result, message] = operate([this, from, n, pos]() { return baseStorage_->write(from, n, pos); });
      file_.write(fmt::format("{} {} {} {} {}\n", kWrite, message, pos, n, result));
      return result;
    }

    IOSize writev(const IOBuffer* from, IOSize n) override {
      auto offset = baseStorage_->position();
      auto const [result, message] = operate([this, from, n]() { return baseStorage_->writev(from, n); });
      std::string elements;
      IOSize total = 0;
      for (IOSize i = 0; i < n; ++i) {
        elements += fmt::format("{} {} {} {}\n", kWritevElement, i, offset, from[i].size());
        total += from[i].size();
        offset += from[i].size();
      }
      file_.write(fmt::format("{} {} {} {} {}\n", kWritev, message, total, result, n) + elements);
      return result;
    }

    IOSize writev(const IOPosBuffer* from, IOSize n) override {
      auto const [result, message] = operate([this, from, n]() { return baseStorage_->writev(from, n); });
      std::string elements;
      IOSize total = 0;
      for (IOSize i = 0; i < n; ++i) {
        elements += fmt::format("{} {} {} {}\n", kWritevElement, i, from[i].offset(), from[i].size());
        total += from[i].size();
      }
      file_.write(fmt::format("{} {} {} {} {}\n", kWritev, message, total, result, n) + elements);
      return result;
    }

    IOOffset position(IOOffset offset, Relative whence) override {
      auto const [result, message] =
          operate([this, offset, whence]() { return baseStorage_->position(offset, whence); });
      file_.write(fmt::format("{} {} {} {}\n", kPosition, message, offset, static_cast<int>(whence)));
      return result;
    }

    void resize(IOOffset size) override {
      auto const message = operate([this, size]() { return baseStorage_->resize(size); });
      file_.write(fmt::format("{} {} {}\n", kResize, message, size));
    }

    void flush() override {
      auto const message = operate([this]() { return baseStorage_->flush(); });
      file_.write(fmt::format("{} {}\n", kFlush, message));
    }

    void close() override {
      auto const message = operate([this]() { return baseStorage_->close(); });
      file_.write(fmt::format("{} {}\n", kClose, message));
    }

    bool prefetch(const IOPosBuffer* what, IOSize n) override {
      auto const [value, message] = operate([this, what, n]() { return baseStorage_->prefetch(what, n); });
      std::string elements;
      IOSize total = 0;
      for (IOSize i = 0; i < n; ++i) {
        elements += fmt::format("{} {} {} {}\n", kPrefetchElement, i, what[i].offset(), what[i].size());
        total += what[i].size();
      }
      file_.write(fmt::format("{} {} {} {} {}\n", kPrefetch, message, total, n, value) + elements);
      return value;
    }

  private:
    template <typename F>
    auto operate(F&& func) -> std::tuple<decltype(func()), std::string> {
      auto const id = idCounter_.fetch_add(1);
      auto const begin = now();
      auto const result = func();
      auto const end = now();
      LogTrace("IOTrace").format("IOTrace {} id {}", traceId_, id);
      return std::tuple(result,
                        fmt::format("{} {} {}",
                                    id,
                                    std::chrono::round<std::chrono::milliseconds>(begin.time_since_epoch()).count(),
                                    std::chrono::round<std::chrono::microseconds>(end - begin).count()));
    }

    template <typename F>
      requires std::is_same_v<std::invoke_result_t<F>, void>
    auto operate(F&& func) -> std::string {
      auto const id = idCounter_.fetch_add(1);
      auto const begin = now();
      func();
      auto const end = now();
      LogTrace("IOTrace").format("IOTrace {} id {}", traceId_, id);
      return fmt::format("{} {} {}",
                         id,
                         std::chrono::round<std::chrono::milliseconds>(begin.time_since_epoch()).count(),
                         std::chrono::round<std::chrono::microseconds>(end - begin).count());
    }

    static std::chrono::time_point<std::chrono::steady_clock> now() { return std::chrono::steady_clock::now(); }

    ThreadSafeOutputFileStream file_;
    std::unique_ptr<Storage> baseStorage_;
    std::atomic<unsigned int> idCounter_{0};
    unsigned int const traceId_;
  };

  class StorageTracerProxyMaker : public StorageProxyMaker {
  public:
    StorageTracerProxyMaker(edm::ParameterSet const& pset)
        : filenamePattern_(pset.getUntrackedParameter<std::string>("traceFilePattern")) {
      if (filenamePattern_.find("%I") == std::string::npos) {
        throw edm::Exception(edm::errors::Configuration) << "traceFilePattern did not contain '%I'";
      }
    }

    static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {
      iDesc.addUntracked<std::string>("traceFilePattern", "trace_%I.txt")
          ->setComment(
              "Pattern for the output trace file names. Must contain '%I' for the counter of different files.");
    }

    std::unique_ptr<Storage> wrap(std::string const& url, std::unique_ptr<Storage> storage) const override {
      auto value = fileCounter_.fetch_add(1);
      std::string fname = filenamePattern_;
      boost::replace_all(fname, "%I", std::to_string(value));
      return std::make_unique<StorageTracerProxy>(value, fname, url, std::move(storage));
    }

  private:
    mutable std::atomic<unsigned int> fileCounter_{0};
    std::string const filenamePattern_;
  };
}  // namespace edm::storage

#include "FWCore/ParameterSet/interface/ValidatedPluginMacros.h"
#include "Utilities/StorageFactory/interface/StorageProxyMakerFactory.h"
DEFINE_EDM_VALIDATED_PLUGIN(edm::storage::StorageProxyMakerFactory,
                            edm::storage::StorageTracerProxyMaker,
                            "StorageTracerProxy");
