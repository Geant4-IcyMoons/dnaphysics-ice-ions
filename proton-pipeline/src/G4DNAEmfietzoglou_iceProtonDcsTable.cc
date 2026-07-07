//
// Table reader and sampler for generated proton ice DCS/TCS files.
//

#include "G4DNAEmfietzoglou_iceProtonDcsTable.hh"

#include "G4Exception.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>

namespace {
std::string WithDatSuffix(const std::string& filename)
{
  if (filename.size() >= 4 &&
      filename.substr(filename.size() - 4) == ".dat") {
    return filename;
  }
  return filename + ".dat";
}

bool FileExists(const std::string& path)
{
  std::ifstream test(path.c_str());
  return test.good();
}

std::string JoinPath(const std::string& directory, const std::string& filename)
{
  if (directory.empty()) return filename;
  if (directory[directory.size() - 1] == '/') return directory + filename;
  return directory + "/" + filename;
}

G4double CrossSectionScale()
{
  return (1.e-22 / 3.343) * m * m;
}
}  // namespace

std::string G4DNAEmfietzoglou_iceProtonDcsTable::ResolvePath(
    const std::string& filename)
{
  const std::string dat = WithDatSuffix(filename);

  std::vector<std::string> candidates;
  const char* protonDir = std::getenv("DNA_PROTON_TABLE_DIR");
  if (protonDir && *protonDir) candidates.push_back(JoinPath(protonDir, dat));

  const char* legacyDir = std::getenv("DNA_ICE_PROTON_DATA_DIR");
  if (legacyDir && *legacyDir) candidates.push_back(JoinPath(legacyDir, dat));

  candidates.push_back(JoinPath("cross_sections", dat));
  candidates.push_back(JoinPath("../cross_sections", dat));
  candidates.push_back(JoinPath("../../cross_sections", dat));

  const char* g4LowEnergy = std::getenv("G4LEDATA");
  if (g4LowEnergy && *g4LowEnergy) {
    candidates.push_back(JoinPath(JoinPath(g4LowEnergy, "dna"), dat));
  }

  for (const auto& candidate : candidates) {
    if (FileExists(candidate)) return candidate;
  }

  std::ostringstream message;
  message << "Missing proton ice cross-section table '" << dat << "'. Tried:";
  for (const auto& candidate : candidates) message << "\n  " << candidate;
  G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::ResolvePath",
              "protondcs001", FatalException, message.str().c_str());
  return dat;
}

void G4DNAEmfietzoglou_iceProtonDcsTable::Load(
    const std::string& totalTableBase,
    const std::string& diffTableFile)
{
  fTotalRows.clear();
  fDiffGrids.clear();
  fNComponents = 0;

  fTotalPath = ResolvePath(totalTableBase);
  fDiffPath = ResolvePath(diffTableFile);

  {
    std::ifstream input(fTotalPath.c_str());
    if (!input) {
      G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                  "protondcs002", FatalException,
                  ("Cannot open total table: " + fTotalPath).c_str());
    }

    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream row(line);
      TotalRow parsed;
      row >> parsed.energy;
      if (!row) continue;
      parsed.energy *= eV;

      G4double value = 0.;
      while (row >> value) parsed.xs.push_back(value * CrossSectionScale());
      if (parsed.xs.empty()) continue;

      if (fNComponents == 0) {
        fNComponents = static_cast<G4int>(parsed.xs.size());
      } else if (fNComponents != static_cast<G4int>(parsed.xs.size())) {
        G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                    "protondcs003", FatalException,
                    "Inconsistent component count in total table.");
      }
      fTotalRows.push_back(parsed);
    }
  }

  std::map<G4double, DiffGrid> grids;
  {
    std::ifstream input(fDiffPath.c_str());
    if (!input) {
      G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                  "protondcs004", FatalException,
                  ("Cannot open DCS table: " + fDiffPath).c_str());
    }

    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream row(line);
      G4double energy = 0.;
      G4double transfer = 0.;
      row >> energy >> transfer;
      if (!row) continue;

      std::vector<G4double> values;
      G4double value = 0.;
      while (row >> value) values.push_back(std::max(0., value));
      if (values.empty()) continue;
      if (fNComponents != static_cast<G4int>(values.size())) {
        G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                    "protondcs005", FatalException,
                    "DCS component count does not match total table.");
      }

      auto& grid = grids[energy * eV];
      grid.energy = energy * eV;
      grid.transfer.push_back(transfer * eV);
      grid.dcs.push_back(values);
    }
  }

  for (auto& item : grids) {
    auto& grid = item.second;
    std::vector<std::size_t> order(grid.transfer.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&grid](std::size_t a, std::size_t b) {
                return grid.transfer[a] < grid.transfer[b];
              });

    DiffGrid sorted;
    sorted.energy = grid.energy;
    sorted.transfer.reserve(order.size());
    sorted.dcs.reserve(order.size());
    for (const auto index : order) {
      sorted.transfer.push_back(grid.transfer[index]);
      sorted.dcs.push_back(grid.dcs[index]);
    }
    fDiffGrids.push_back(sorted);
  }

  std::sort(fTotalRows.begin(), fTotalRows.end(),
            [](const TotalRow& a, const TotalRow& b) {
              return a.energy < b.energy;
            });
  std::sort(fDiffGrids.begin(), fDiffGrids.end(),
            [](const DiffGrid& a, const DiffGrid& b) {
              return a.energy < b.energy;
            });

  if (fTotalRows.empty() || fDiffGrids.empty() || fNComponents <= 0) {
    G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                "protondcs006", FatalException,
                "Empty or malformed proton ice table.");
  }
}

G4int G4DNAEmfietzoglou_iceProtonDcsTable::LowerIndex(
    const std::vector<G4double>& values,
    G4double x)
{
  if (values.size() < 2 || x <= values.front()) return 0;
  if (x >= values.back()) return static_cast<G4int>(values.size()) - 2;
  auto upper = std::upper_bound(values.begin(), values.end(), x);
  return static_cast<G4int>(std::distance(values.begin(), upper)) - 1;
}

G4double G4DNAEmfietzoglou_iceProtonDcsTable::Interpolate(
    G4double x1,
    G4double x2,
    G4double x,
    G4double y1,
    G4double y2)
{
  if (x2 == x1) return y1;
  if (y1 > 0. && y2 > 0. && x1 > 0. && x2 > 0. && x > 0.) {
    const G4double f =
        (std::log(x) - std::log(x1)) / (std::log(x2) - std::log(x1));
    return std::exp(std::log(y1) + f * (std::log(y2) - std::log(y1)));
  }
  return std::max(0., y1 + (x - x1) * (y2 - y1) / (x2 - x1));
}

G4double G4DNAEmfietzoglou_iceProtonDcsTable::TotalCrossSection(
    G4double kineticEnergy,
    G4int component) const
{
  if (fTotalRows.empty() ||
      kineticEnergy < fTotalRows.front().energy ||
      kineticEnergy > fTotalRows.back().energy) {
    return 0.;
  }
  if (component >= fNComponents) return 0.;

  std::vector<G4double> energies;
  energies.reserve(fTotalRows.size());
  for (const auto& row : fTotalRows) energies.push_back(row.energy);
  const G4int index = LowerIndex(energies, kineticEnergy);
  const auto& lo = fTotalRows[index];
  const auto& hi = fTotalRows[index + 1];

  auto componentValue = [&](G4int c) {
    return Interpolate(lo.energy, hi.energy, kineticEnergy, lo.xs[c], hi.xs[c]);
  };

  if (component >= 0) return componentValue(component);

  G4double total = 0.;
  for (G4int c = 0; c < fNComponents; ++c) total += componentValue(c);
  return total;
}

G4int G4DNAEmfietzoglou_iceProtonDcsTable::SelectComponent(
    G4double kineticEnergy) const
{
  G4double total = 0.;
  std::vector<G4double> values(fNComponents, 0.);
  for (G4int c = 0; c < fNComponents; ++c) {
    values[c] = TotalCrossSection(kineticEnergy, c);
    total += values[c];
  }
  if (total <= 0.) return 0;

  G4double pick = G4UniformRand() * total;
  for (G4int c = 0; c < fNComponents; ++c) {
    if (pick <= values[c]) return c;
    pick -= values[c];
  }
  return fNComponents - 1;
}

std::vector<G4double> G4DNAEmfietzoglou_iceProtonDcsTable::InterpolatedDcs(
    G4double kineticEnergy,
    G4int component) const
{
  if (fDiffGrids.empty() || component < 0 || component >= fNComponents) {
    return {};
  }

  std::vector<G4double> energies;
  energies.reserve(fDiffGrids.size());
  for (const auto& grid : fDiffGrids) energies.push_back(grid.energy);
  const G4int index = LowerIndex(energies, kineticEnergy);
  const auto& lo = fDiffGrids[index];
  const auto& hi = fDiffGrids[index + 1];

  if (lo.transfer.size() != hi.transfer.size()) {
    const auto& nearest =
        std::abs(kineticEnergy - lo.energy) <= std::abs(hi.energy - kineticEnergy)
            ? lo
            : hi;
    std::vector<G4double> values;
    values.reserve(nearest.dcs.size());
    for (const auto& row : nearest.dcs) values.push_back(row[component]);
    return values;
  }

  std::vector<G4double> values;
  values.reserve(lo.dcs.size());
  for (std::size_t i = 0; i < lo.dcs.size(); ++i) {
    if (std::abs(lo.transfer[i] - hi.transfer[i]) >
        1.e-9 * std::max(lo.transfer[i], hi.transfer[i])) {
      const auto& nearest =
          std::abs(kineticEnergy - lo.energy) <= std::abs(hi.energy - kineticEnergy)
              ? lo
              : hi;
      values.clear();
      values.reserve(nearest.dcs.size());
      for (const auto& row : nearest.dcs) values.push_back(row[component]);
      return values;
    }
    values.push_back(Interpolate(lo.energy,
                                 hi.energy,
                                 kineticEnergy,
                                 lo.dcs[i][component],
                                 hi.dcs[i][component]));
  }
  return values;
}

G4double G4DNAEmfietzoglou_iceProtonDcsTable::SampleTransferEnergy(
    G4double kineticEnergy,
    G4int component) const
{
  if (fDiffGrids.empty() || component < 0 || component >= fNComponents) {
    return 0.;
  }

  std::vector<G4double> energies;
  energies.reserve(fDiffGrids.size());
  for (const auto& grid : fDiffGrids) energies.push_back(grid.energy);
  const G4int index = LowerIndex(energies, kineticEnergy);
  const auto& baseGrid = fDiffGrids[index];
  const auto dcs = InterpolatedDcs(kineticEnergy, component);
  if (baseGrid.transfer.size() < 2 || dcs.size() != baseGrid.transfer.size()) {
    return 0.;
  }

  G4double area = 0.;
  for (std::size_t i = 1; i < baseGrid.transfer.size(); ++i) {
    const G4double y0 = std::max(0., dcs[i - 1]);
    const G4double y1 = std::max(0., dcs[i]);
    const G4double dx = baseGrid.transfer[i] - baseGrid.transfer[i - 1];
    if (dx > 0.) area += 0.5 * (y0 + y1) * dx;
  }
  if (area <= 0.) return 0.;

  G4double pick = G4UniformRand() * area;
  for (std::size_t i = 1; i < baseGrid.transfer.size(); ++i) {
    const G4double y0 = std::max(0., dcs[i - 1]);
    const G4double y1 = std::max(0., dcs[i]);
    const G4double x0 = baseGrid.transfer[i - 1];
    const G4double x1 = baseGrid.transfer[i];
    const G4double dx = x1 - x0;
    if (dx <= 0.) continue;
    const G4double segment = 0.5 * (y0 + y1) * dx;
    if (pick <= segment) {
      const G4double fraction = segment > 0. ? pick / segment : 0.;
      return x0 + fraction * dx;
    }
    pick -= segment;
  }
  return baseGrid.transfer.back();
}
