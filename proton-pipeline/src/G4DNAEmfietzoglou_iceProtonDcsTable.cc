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
    const std::string& diffTableFile, G4double projectileMassEnergy)
{
  fTotalRows.clear();
  fDiffGrids.clear();
  fNComponents = 0;
  fEnergies.clear();
  fProjectileMassEnergy = projectileMassEnergy;
  if (!std::isfinite(projectileMassEnergy) || projectileMassEnergy <= 0.) {
    G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                "protondcs007", FatalException, "Projectile rest energy must be positive.");
  }

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
      if (!row.eof()) {
        G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                    "protondcs013", FatalException, "Malformed TCS numeric field.");
      }
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
      if (!std::isfinite(energy) || energy <= 0.
          || !std::isfinite(transfer) || transfer <= 0.) {
        G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                    "protondcs014", FatalException, "Invalid DCS energy.");
      }

      std::vector<G4double> values;
      G4double value = 0.;
      while (row >> value) {
        if (!std::isfinite(value) || value < 0.) {
          G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                      "protondcs008", FatalException, "Negative or non-finite DCS.");
        }
        values.push_back(value);
      }
      if (!row.eof()) {
        G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                    "protondcs015", FatalException, "Malformed DCS numeric field.");
      }
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
  if (fTotalRows.size() != fDiffGrids.size()) {
    G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                "protondcs009", FatalException, "TCS/DCS incident grids differ; regenerate tables.");
  }
  for (std::size_t i = 0; i < fDiffGrids.size(); ++i) {
    auto& grid = fDiffGrids[i];
    if (!std::isfinite(grid.energy) || grid.energy <= 0.
        || grid.energy != fTotalRows[i].energy || grid.transfer.size() < 2
        || (i > 0 && grid.energy <= fDiffGrids[i-1].energy)) {
      G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                  "protondcs010", FatalException, "Invalid or mismatched DCS/TCS grid.");
    }
    fEnergies.push_back(grid.energy);
    grid.cumulative.assign(fNComponents, std::vector<G4double>(grid.transfer.size(), 0.));
    for (std::size_t j = 0; j < grid.transfer.size(); ++j) {
      if (!std::isfinite(grid.transfer[j]) || grid.transfer[j] <= 0.
          || (j > 0 && grid.transfer[j] <= grid.transfer[j-1])) {
        G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                    "protondcs011", FatalException, "Invalid transfer-energy grid.");
      }
      if (j == 0) continue;
      for (G4int c = 0; c < fNComponents; ++c) {
        grid.cumulative[c][j] = grid.cumulative[c][j-1]
            + 0.5 * (grid.dcs[j-1][c] + grid.dcs[j][c])
            * ((grid.transfer[j] - grid.transfer[j-1]) / eV);
      }
    }
    for (G4int c = 0; c < fNComponents; ++c) {
      const G4double integrated = grid.cumulative[c].back() * CrossSectionScale();
      const G4double stored = fTotalRows[i].xs[c];
      if (!std::isfinite(integrated) || !std::isfinite(stored) || stored < 0.
          || std::abs(stored - integrated) > 1.e-7 * std::max(stored, integrated)) {
        G4Exception("G4DNAEmfietzoglou_iceProtonDcsTable::Load",
                    "protondcs012", FatalException,
                    "TCS is not the piecewise-linear DCS integral; regenerate tables.");
      }
    }
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

G4double G4DNAEmfietzoglou_iceProtonDcsTable::MaximumTransfer(G4double kineticEnergy) const
{
  // Same high-mass cutoff as the generator; no recoil-denominator change.
  const G4double tau = kineticEnergy / fProjectileMassEnergy;
  return std::min(kineticEnergy, 2. * (510998.95 * eV) * tau * (tau + 2.));
}

G4double G4DNAEmfietzoglou_iceProtonDcsTable::UpperWeight(
    G4int index, G4double kineticEnergy) const
{
  if (fEnergies.size() == 1) return 0.;
  return std::clamp(std::log(kineticEnergy / fEnergies[index])
                    / std::log(fEnergies[index+1] / fEnergies[index]), 0., 1.);
}

G4double G4DNAEmfietzoglou_iceProtonDcsTable::AreaBelow(
    const DiffGrid& grid, G4int component, G4double upper)
{
  if (upper <= grid.transfer.front()) return 0.;
  if (upper >= grid.transfer.back()) return grid.cumulative[component].back();
  const G4int i = LowerIndex(grid.transfer, upper);
  const G4double fraction = (upper - grid.transfer[i])
                           / (grid.transfer[i+1] - grid.transfer[i]);
  const G4double y0 = grid.dcs[i][component];
  const G4double y = y0 + fraction * (grid.dcs[i+1][component] - y0);
  return grid.cumulative[component][i]
      + 0.5 * (y0 + y) * ((upper - grid.transfer[i]) / eV);
}

G4double G4DNAEmfietzoglou_iceProtonDcsTable::SampleArea(
    const DiffGrid& grid, G4int component, G4double area)
{
  const auto& cumulative = grid.cumulative[component];
  const auto end = std::upper_bound(cumulative.begin(), cumulative.end(), area);
  if (end == cumulative.end()) return grid.transfer.back();
  const std::size_t i = std::distance(cumulative.begin(), end) - 1;
  const G4double dx = grid.transfer[i+1] - grid.transfer[i];
  const G4double y0 = grid.dcs[i][component];
  const G4double delta = grid.dcs[i+1][component] - y0;
  const G4double a = std::max(0., area - cumulative[i]) / (dx / eV);
  // Invert y0*u + (y1-y0)*u^2/2 = a, without subtractive cancellation.
  const G4double denominator = y0 + std::sqrt(std::max(0., y0*y0 + 2.*delta*a));
  const G4double u = denominator > 0. ? 2.*a / denominator : 0.;
  return grid.transfer[i] + std::clamp(u, 0., 1.) * dx;
}

G4double G4DNAEmfietzoglou_iceProtonDcsTable::TotalCrossSection(
    G4double kineticEnergy, G4int component) const
{
  if (fEnergies.empty() || kineticEnergy < fEnergies.front()
      || kineticEnergy > fEnergies.back() || component >= fNComponents) return 0.;
  const G4int lo = LowerIndex(fEnergies, kineticEnergy);
  const G4int hi = fEnergies.size() == 1 ? lo : lo + 1;
  const G4double weight = UpperWeight(lo, kineticEnergy);
  const G4double upper = MaximumTransfer(kineticEnergy);
  G4double total = 0.;
  for (G4int c = (component < 0 ? 0 : component);
       c < (component < 0 ? fNComponents : component + 1); ++c) {
    total += (1. - weight) * AreaBelow(fDiffGrids[lo], c, upper)
            + weight * AreaBelow(fDiffGrids[hi], c, upper);
  }
  return total * CrossSectionScale();
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

G4double G4DNAEmfietzoglou_iceProtonDcsTable::SampleTransferEnergy(
    G4double kineticEnergy, G4int component) const
{
  if (fEnergies.empty() || component < 0 || component >= fNComponents
      || kineticEnergy < fEnergies.front() || kineticEnergy > fEnergies.back()) return 0.;
  const G4int lo = LowerIndex(fEnergies, kineticEnergy);
  const G4int hi = fEnergies.size() == 1 ? lo : lo + 1;
  const G4double weight = UpperWeight(lo, kineticEnergy);
  const G4double upper = MaximumTransfer(kineticEnergy);
  const G4double loArea = (1. - weight) * AreaBelow(fDiffGrids[lo], component, upper);
  const G4double hiArea = weight * AreaBelow(fDiffGrids[hi], component, upper);
  if (loArea + hiArea <= 0.) return 0.;
  const G4double pick = G4UniformRand() * (loArea + hiArea);
  // Convex interpolation of DCS at fixed W, linear in log(T). A mixture
  // samples this exactly even when the neighbouring W grids differ.
  const G4double sampled = pick < loArea
      ? SampleArea(fDiffGrids[lo], component, pick / (1. - weight))
      : SampleArea(fDiffGrids[hi], component, (pick - loArea) / weight);
  return std::min(sampled, upper);
}
