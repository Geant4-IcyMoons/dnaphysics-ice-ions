// ModelDataRegistry.hh
#pragma once

#include <map>
#include <mutex>
#include <string>

// Registry for data tables actually loaded by models during a run.
// This avoids hardcoding reference filenames in RunAction.
class ModelDataRegistry {
public:
  static ModelDataRegistry& Instance();

  void Clear();
  void Record(const std::string& key, const std::string& value);
  std::map<std::string, std::string> Snapshot() const;

  // Normalize a path-like string to a basename ending in ".dat".
  static std::string NormalizeDatBasename(const std::string& pathLike);

private:
  ModelDataRegistry() = default;
  mutable std::mutex fMutex;
  std::map<std::string, std::string> fData;
};
