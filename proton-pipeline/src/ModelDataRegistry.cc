// ModelDataRegistry.cc

#include "ModelDataRegistry.hh"

ModelDataRegistry& ModelDataRegistry::Instance()
{
  static ModelDataRegistry instance;
  return instance;
}

void ModelDataRegistry::Clear()
{
  std::lock_guard<std::mutex> lock(fMutex);
  fData.clear();
}

void ModelDataRegistry::Record(const std::string& key, const std::string& value)
{
  if (key.empty() || value.empty()) {
    return;
  }
  std::lock_guard<std::mutex> lock(fMutex);
  if (fData.find(key) != fData.end()) {
    return;
  }
  fData.emplace(key, value);
}

std::map<std::string, std::string> ModelDataRegistry::Snapshot() const
{
  std::lock_guard<std::mutex> lock(fMutex);
  return fData;
}

std::string ModelDataRegistry::NormalizeDatBasename(const std::string& pathLike)
{
  if (pathLike.empty()) {
    return {};
  }
  std::string base = pathLike;
  const std::string::size_type slash = base.find_last_of("/\\");
  if (slash != std::string::npos) {
    base = base.substr(slash + 1);
  }
  if (base.size() < 4 || base.substr(base.size() - 4) != ".dat") {
    base += ".dat";
  }
  return base;
}
