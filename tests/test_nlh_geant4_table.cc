#include "G4DNANLHHardElastic.hh"
#include "G4DNANLHHardElasticTable.hh"

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4SystemOfUnits.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using Table = G4DNANLHHardElasticTable;

void RequireClose(const std::string& label,
                  G4double actual,
                  G4double expected,
                  G4double relativeTolerance = 5.e-13,
                  G4double absoluteTolerance = 1.e-14)
{
  const G4double error = std::abs(actual - expected);
  const G4double limit =
      std::max(absoluteTolerance, relativeTolerance * std::abs(expected));
  if (error > limit) {
    throw std::runtime_error(label + " differs: actual=" +
                             std::to_string(actual) + ", expected=" +
                             std::to_string(expected));
  }
}
}  // namespace

int main(int argc, char** argv)
{
  try {
    if (argc != 2) throw std::runtime_error("Expected the carbon table path.");
    const Table table(argv[1]);
    if (table.ReleaseStatus() != "atomistic_validation_pending") {
      throw std::runtime_error("Unexpected release status.");
    }

    struct CrossSectionProbe
    {
      Table::Target target;
      G4double energyEV;
      G4double expectedAngstrom2;
    };
    const CrossSectionProbe crossSectionProbes[] = {
        {Table::Target::Hydrogen, 1100., 0.41261190967809536},
        {Table::Target::Hydrogen, 37000., 0.63008751847239919},
        {Table::Target::Hydrogen, 2.3e6, 0.63664391950149113},
        {Table::Target::Hydrogen, 8.9e7, 0.6367483462538146},
        {Table::Target::Oxygen, 1100., 1.4552558920821328},
        {Table::Target::Oxygen, 37000., 1.5260340900551848},
        {Table::Target::Oxygen, 2.3e6, 1.5281678937153049},
        {Table::Target::Oxygen, 8.9e7, 1.5282018797680692},
    };
    for (const auto& probe : crossSectionProbes) {
      RequireClose("hard cross section",
                   table.HardCrossSection(probe.target, probe.energyEV * eV) /
                       (angstrom * angstrom),
                   probe.expectedAngstrom2);
    }
    RequireClose("below table", table.HardCrossSection(
                                        Table::Target::Hydrogen, 999. * eV), 0.);
    RequireClose("above table", table.HardCrossSection(
                                        Table::Target::Oxygen, 100.1 * MeV), 0.);

    struct AngleProbe
    {
      Table::Target target;
      G4double energyEV;
      G4double quantile;
      G4double theta;
      G4double recoilEV;
    };
    const AngleProbe angleProbes[] = {
        {Table::Target::Hydrogen, 1100., .137, 1.8985032124256291,
         207.85925895599414},
        {Table::Target::Hydrogen, 37000., .5, 0.044055585063224115,
         5.1320508755817613},
        {Table::Target::Hydrogen, 2.3e6, .913, 0.00037504653553365082,
         0.023125283485781942},
        {Table::Target::Hydrogen, 8.9e7, .137, 5.3995516958998443e-05,
         0.018599185004042548},
        {Table::Target::Oxygen, 1100., .137, 1.0488245160807732,
         270.15950408142152},
        {Table::Target::Oxygen, 37000., .5, 0.012139091272805413,
         1.3352839750220757},
        {Table::Target::Oxygen, 2.3e6, .913, 7.0083038187036039e-05,
         0.002766683387362415},
        {Table::Target::Oxygen, 8.9e7, .137, 2.4534153552047006e-05,
         0.013121157307474833},
    };
    for (const auto& probe : angleProbes) {
      const G4double energy = probe.energyEV * eV;
      const G4double theta =
          table.ThetaCM(probe.target, energy, probe.quantile);
      RequireClose("theta_cm", theta, probe.theta, 2.e-12, 1.e-15);
      const auto outcome = table.Outcome(probe.target, energy, theta);
      RequireClose("recoil", outcome.recoilEnergy / eV, probe.recoilEV,
                   2.e-12, 1.e-12);
      RequireClose("kinetic-energy conservation",
                   (outcome.projectileEnergy + outcome.recoilEnergy) / eV,
                   probe.energyEV, 2.e-14, 1.e-9);
    }

    setenv("DNA_NLH_HARD_ELASTIC_DATA", argv[1], 1);
    setenv("DNA_NLH_ALLOW_VALIDATION_PENDING", "1", 1);
    auto* nist = G4NistManager::Instance();
    G4Material* water = nist->FindOrBuildMaterial("G4_WATER");
    G4Material* hex = nist->BuildMaterialWithNewDensity(
        "NLH_TEST_HEX", "G4_WATER", 0.9335 * g / cm3);
    G4Material* amorphous = nist->BuildMaterialWithNewDensity(
        "NLH_TEST_AM", "G4_WATER", 0.94 * g / cm3);
    G4DNANLHHardElastic process;
    const G4double energy = 1. * MeV;
    const G4double waterRate = process.MacroscopicCrossSection(water, energy);
    const G4double hexRate = process.MacroscopicCrossSection(hex, energy);
    const G4double amorphousRate =
        process.MacroscopicCrossSection(amorphous, energy);
    RequireClose("hexagonal density scaling", hexRate / waterRate, 0.9335,
                 2.e-12);
    RequireClose("amorphous density scaling", amorphousRate / waterRate, 0.94,
                 2.e-12);
    RequireClose("phase density ratio", amorphousRate / hexRate,
                 0.94 / 0.9335, 2.e-12);

    std::cout << "Carbon NLH Geant4 table/process validation passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
