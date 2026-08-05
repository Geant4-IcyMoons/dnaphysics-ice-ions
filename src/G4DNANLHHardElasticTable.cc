#include "G4DNANLHHardElasticTable.hh"

#include "G4Exception.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace {
[[noreturn]] void TableError(const std::string& message)
{
  G4ExceptionDescription description;
  description << message;
  G4Exception("G4DNANLHHardElasticTable", "nlhTable001", FatalException,
              description);
  throw std::runtime_error(message);
}

G4double ParseFinitePositive(const std::map<std::string, std::string>& metadata,
                             const std::string& key)
{
  const auto found = metadata.find(key);
  if (found == metadata.end()) TableError("Missing NLH table metadata: " + key);
  char* end = nullptr;
  const G4double value = std::strtod(found->second.c_str(), &end);
  if (end == found->second.c_str() || *end != '\0' || !std::isfinite(value) ||
      value <= 0.) {
    TableError("Invalid NLH table metadata: " + key);
  }
  return value;
}

G4bool StrictlyIncreasing(const std::vector<G4double>& values)
{
  for (std::size_t index = 1; index < values.size(); ++index) {
    if (!(values[index] > values[index - 1])) return false;
  }
  return true;
}

G4bool StrictlyDecreasing(const std::vector<G4double>& values)
{
  for (std::size_t index = 1; index < values.size(); ++index) {
    if (!(values[index] < values[index - 1])) return false;
  }
  return true;
}
}  // namespace

std::size_t G4DNANLHHardElasticTable::Index(Target target)
{
  return static_cast<std::size_t>(target);
}

const G4DNANLHHardElasticTable::PairTable&
G4DNANLHHardElasticTable::Pair(Target target) const
{
  return fPairs[Index(target)];
}

G4DNANLHHardElasticTable::G4DNANLHHardElasticTable(const std::string& path)
  : fPath(path)
{
  std::ifstream input(path);
  if (!input) TableError("Could not open NLH hard-elastic table " + path);

  std::map<std::string, std::string> metadata;
  std::array<std::map<G4double, EnergyNode>, 2> grouped;
  std::string line;
  std::size_t lineNumber = 0;
  while (std::getline(input, line)) {
    ++lineNumber;
    if (line.empty()) continue;
    if (line.rfind("# ", 0) == 0) {
      std::istringstream stream(line.substr(2));
      std::string key;
      stream >> key;
      std::string value;
      std::getline(stream >> std::ws, value);
      if (!key.empty() && !value.empty()) metadata.emplace(key, value);
      continue;
    }

    std::istringstream stream(line);
    std::string targetName;
    G4double energyEV = 0.;
    G4double quantile = 0.;
    G4double theta = 0.;
    std::string remainder;
    if (!(stream >> targetName >> energyEV >> quantile >> theta) ||
        (stream >> remainder) || !std::isfinite(energyEV) || energyEV <= 0. ||
        !std::isfinite(quantile) || !std::isfinite(theta)) {
      TableError("Malformed NLH data row at line " + std::to_string(lineNumber));
    }
    Target target;
    if (targetName == "H") {
      target = Target::Hydrogen;
    } else if (targetName == "O") {
      target = Target::Oxygen;
    } else {
      TableError("Unsupported NLH target at line " + std::to_string(lineNumber));
    }
    const G4double energy = energyEV * eV;
    auto& node = grouped[Index(target)][energy];
    node.energy = energy;
    node.quantiles.push_back(quantile);
    node.thetaCM.push_back(theta);
  }

  const auto schema = metadata.find("nlh_geant4_hard_elastic_table_schema");
  if (schema == metadata.end() || schema->second != "1") {
    TableError("Only NLH Geant4 table schema 1 is supported.");
  }
  const auto projectile = metadata.find("projectile");
  if (projectile == metadata.end() || projectile->second != "C") {
    TableError("The released Geant4 hard-elastic table must be for carbon.");
  }
  fProjectile = projectile->second;
  const auto releaseStatus = metadata.find("release_status");
  if (releaseStatus == metadata.end() ||
      (releaseStatus->second != "atomistic_validation_pending" &&
       releaseStatus->second != "accepted")) {
    TableError("Missing or invalid NLH release_status metadata.");
  }
  fReleaseStatus = releaseStatus->second;
  fProjectileMass = ParseFinitePositive(metadata, "projectile_mass_c2_eV") * eV;
  fMinimumTurningPotential =
      ParseFinitePositive(metadata, "minimum_turning_potential_eV") * eV;
  const auto sourceSha = metadata.find("source_csv_sha256");
  if (sourceSha == metadata.end() || sourceSha->second.size() != 64) {
    TableError("Missing or invalid source_csv_sha256 metadata.");
  }
  fSourceCsvSha256 = sourceSha->second;

  for (const auto target : {Target::Hydrogen, Target::Oxygen}) {
    const std::string symbol = target == Target::Hydrogen ? "H" : "O";
    PairTable& pair = fPairs[Index(target)];
    pair.targetMass =
        ParseFinitePositive(metadata, "target_" + symbol + "_mass_c2_eV") * eV;
    pair.thresholdRadius = ParseFinitePositive(
                               metadata,
                               "target_" + symbol +
                                   "_threshold_radius_angstrom") *
                           angstrom;
    pair.energyMinimum =
        ParseFinitePositive(metadata, "target_" + symbol + "_energy_min_eV") *
        eV;
    pair.energyMaximum =
        ParseFinitePositive(metadata, "target_" + symbol + "_energy_max_eV") *
        eV;
    for (auto& entry : grouped[Index(target)]) {
      pair.nodes.push_back(std::move(entry.second));
    }
  }
  Validate();
}

void G4DNANLHHardElasticTable::Validate() const
{
  if (fMinimumTurningPotential < 10. * eV) {
    TableError("The NLH turning threshold cannot be below the published 10 eV domain.");
  }
  for (const auto target : {Target::Hydrogen, Target::Oxygen}) {
    const PairTable& pair = Pair(target);
    if (pair.nodes.size() < 2 || pair.energyMinimum >= pair.energyMaximum ||
        pair.nodes.front().energy != pair.energyMinimum ||
        pair.nodes.back().energy != pair.energyMaximum) {
      TableError("Invalid NLH energy mesh or declared bounds.");
    }
    G4double previousEnergy = -1.;
    for (const auto& node : pair.nodes) {
      if (!(node.energy > previousEnergy) || node.quantiles.size() < 2 ||
          node.quantiles.size() != node.thetaCM.size() ||
          node.quantiles.front() != 0. || node.quantiles.back() != 1. ||
          !StrictlyIncreasing(node.quantiles) || !StrictlyDecreasing(node.thetaCM)) {
        TableError("Invalid NLH impact-parameter mesh.");
      }
      for (const G4double theta : node.thetaCM) {
        if (!std::isfinite(theta) || theta < 0. || theta > pi) {
          TableError("NLH CM angle lies outside [0, pi].");
        }
      }
      previousEnergy = node.energy;
    }
  }
}

G4bool G4DNANLHHardElasticTable::InEnergyRange(
    Target target, G4double projectileEnergy) const
{
  const PairTable& pair = Pair(target);
  return std::isfinite(projectileEnergy) &&
         projectileEnergy >= pair.energyMinimum &&
         projectileEnergy <= pair.energyMaximum;
}

G4double G4DNANLHHardElasticTable::EnergyMinimum(Target target) const
{
  return Pair(target).energyMinimum;
}

G4double G4DNANLHHardElasticTable::EnergyMaximum(Target target) const
{
  return Pair(target).energyMaximum;
}

G4DNANLHHardElasticTable::PairKinematics
G4DNANLHHardElasticTable::Kinematics(Target target,
                                     G4double projectileEnergy) const
{
  const G4double mass1 = fProjectileMass;
  const G4double mass2 = Pair(target).targetMass;
  const G4double incomingMomentum =
      std::sqrt(projectileEnergy * (projectileEnergy + 2. * mass1));
  const G4double invariantMass =
      std::hypot(mass1 + mass2, std::sqrt(2. * mass2 * projectileEnergy));
  const G4double betaCM =
      incomingMomentum / (mass1 + projectileEnergy + mass2);
  PairKinematics result;
  result.invariantMass = invariantMass;
  result.relativeKineticEnergy =
      2. * mass2 * projectileEnergy / (invariantMass + mass1 + mass2);
  result.incomingMomentum = incomingMomentum;
  result.momentumCM = mass2 * incomingMomentum / invariantMass;
  result.betaCM = betaCM;
  result.gammaCM = 1. / std::sqrt((1. - betaCM) * (1. + betaCM));
  return result;
}

G4double G4DNANLHHardElasticTable::HardCrossSection(
    Target target, G4double projectileEnergy) const
{
  if (!InEnergyRange(target, projectileEnergy)) return 0.;
  const PairTable& pair = Pair(target);
  const G4double energyCM = Kinematics(target, projectileEnergy).relativeKineticEnergy;
  if (energyCM <= fMinimumTurningPotential) return 0.;
  return pi * pair.thresholdRadius * pair.thresholdRadius *
         (1. - fMinimumTurningPotential / energyCM);
}

G4double G4DNANLHHardElasticTable::AngleAtQuantile(const EnergyNode& node,
                                                   G4double quantile)
{
  if (quantile <= 0.) return node.thetaCM.front();
  if (quantile >= 1.) return node.thetaCM.back();
  const auto upper =
      std::lower_bound(node.quantiles.begin(), node.quantiles.end(), quantile);
  const std::size_t upperIndex =
      static_cast<std::size_t>(upper - node.quantiles.begin());
  if (*upper == quantile) return node.thetaCM[upperIndex];
  const std::size_t lowerIndex = upperIndex - 1;
  const G4double fraction =
      (quantile - node.quantiles[lowerIndex]) /
      (node.quantiles[upperIndex] - node.quantiles[lowerIndex]);
  return node.thetaCM[lowerIndex] +
         fraction * (node.thetaCM[upperIndex] - node.thetaCM[lowerIndex]);
}

G4double G4DNANLHHardElasticTable::ThetaCM(Target target,
                                           G4double projectileEnergy,
                                           G4double areaQuantile) const
{
  if (!InEnergyRange(target, projectileEnergy)) {
    TableError("NLH angular interpolation requested outside its energy range.");
  }
  if (!std::isfinite(areaQuantile) || areaQuantile < 0. || areaQuantile > 1.) {
    TableError("NLH collision-area quantile must lie in [0, 1].");
  }
  const auto& nodes = Pair(target).nodes;
  const auto upper = std::lower_bound(
      nodes.begin(), nodes.end(), projectileEnergy,
      [](const EnergyNode& node, G4double energy) { return node.energy < energy; });
  if (upper != nodes.end() && upper->energy == projectileEnergy) {
    return AngleAtQuantile(*upper, areaQuantile);
  }
  if (upper == nodes.begin() || upper == nodes.end()) {
    TableError("NLH energy interpolation could not bracket the requested energy.");
  }
  const auto& lower = *(upper - 1);
  const G4double thetaLower = AngleAtQuantile(lower, areaQuantile);
  const G4double thetaUpper = AngleAtQuantile(*upper, areaQuantile);
  const G4double fraction =
      std::log(projectileEnergy / lower.energy) /
      std::log(upper->energy / lower.energy);
  if (thetaLower > 0. && thetaUpper > 0.) {
    return std::exp((1. - fraction) * std::log(thetaLower) +
                    fraction * std::log(thetaUpper));
  }
  return (1. - fraction) * thetaLower + fraction * thetaUpper;
}

G4DNANLHHardElasticTable::TwoBodyOutcome
G4DNANLHHardElasticTable::Outcome(Target target,
                                  G4double projectileEnergy,
                                  G4double thetaCM) const
{
  if (!InEnergyRange(target, projectileEnergy) || !std::isfinite(thetaCM) ||
      thetaCM < 0. || thetaCM > pi) {
    TableError("Invalid NLH two-body outcome request.");
  }
  const PairKinematics kinematics = Kinematics(target, projectileEnergy);
  const G4double energy1CM =
      std::hypot(fProjectileMass, kinematics.momentumCM);
  TwoBodyOutcome result;
  result.projectileTransverseMomentum =
      kinematics.momentumCM * std::sin(thetaCM);
  result.projectileLongitudinalMomentum = kinematics.gammaCM *
      (kinematics.momentumCM * std::cos(thetaCM) +
       kinematics.betaCM * energy1CM);
  result.recoilTransverseMomentum = -result.projectileTransverseMomentum;
  result.recoilLongitudinalMomentum =
      kinematics.incomingMomentum - result.projectileLongitudinalMomentum;
  result.recoilEnergy =
      kinematics.momentumCM * kinematics.momentumCM *
      (1. - std::cos(thetaCM)) / Pair(target).targetMass;
  const G4double tolerance = 1.e-12 * std::max(1. * eV, projectileEnergy);
  if (result.recoilEnergy < -tolerance ||
      result.recoilEnergy > projectileEnergy + tolerance) {
    TableError("NLH recoil energy lies outside its physical bounds.");
  }
  result.recoilEnergy =
      std::min(projectileEnergy, std::max(0., result.recoilEnergy));
  result.projectileEnergy = projectileEnergy - result.recoilEnergy;
  return result;
}
