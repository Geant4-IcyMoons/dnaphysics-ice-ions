#include "G4DNAZBLFullElastic.hh"

#include "G4GenericIon.hh"
#include "G4IonConstructor.hh"
#include "G4IonTable.hh"
#include "G4NistManager.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4ProcessManager.hh"
#include "G4Proton.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void Require(G4bool condition, const std::string& message)
{
  if (!condition) throw std::runtime_error(message);
}

void RequireClose(const std::string& label, G4double value,
                  G4double expected, G4double relativeTolerance)
{
  const G4double scale = std::max(std::abs(expected), 1.e-300);
  if (std::abs(value - expected) > relativeTolerance * scale) {
    throw std::runtime_error(label + " mismatch");
  }
}
}  // namespace

int main()
{
  try {
    RequireClose("phi(0)", G4DNAZBLFullElastic::Screening(0.),
                 1.00007, 1.e-12);
    RequireClose("phi(1)", G4DNAZBLFullElastic::Screening(1.),
                 0.4164406620, 1.e-9);

    auto* genericIon = G4GenericIon::GenericIonDefinition();
    genericIon->SetProcessManager(new G4ProcessManager(genericIon));
    G4IonConstructor constructor;
    constructor.ConstructParticle();
    G4ParticleTable::GetParticleTable()->SetReadiness();
    auto* carbon = G4IonTable::GetIonTable()->GetIon(6, 12, 0.);
    auto* oxygen = G4IonTable::GetIonTable()->GetIon(8, 16, 0.);
    Require(carbon && oxygen, "Could not construct C-12/O-16 definitions");

    const G4double sigmaCO = G4DNAZBLFullElastic::PairCrossSection(
        6, carbon->GetPDGMass(), 8, oxygen->GetPDGMass(), 1. * MeV, 10. * eV);
    Require(std::isfinite(sigmaCO) && sigmaCO > 0.,
            "C--O ZBL cross section is not positive and finite");
    RequireClose("Python/C++ C--O reference", sigmaCO / (angstrom * angstrom),
                 0.04936017, 2.e-3);
    const G4double sigmaCO1 = G4DNAZBLFullElastic::PairCrossSection(
        6, carbon->GetPDGMass(), 8, oxygen->GetPDGMass(), 1. * MeV, 1. * eV);
    const G4double sigmaCO30 = G4DNAZBLFullElastic::PairCrossSection(
        6, carbon->GetPDGMass(), 8, oxygen->GetPDGMass(), 1. * MeV, 30. * eV);
    Require(sigmaCO1 > sigmaCO && sigmaCO > sigmaCO30,
            "ZBL cross section does not decrease with transfer cutoff");

    const G4double length = G4DNAZBLFullElastic::ScreeningLength(6, 8);
    G4double previous = -1.;
    for (const G4double factor : {0., 0.5, 1., 2., 4.}) {
      const G4double cosine = G4DNAZBLFullElastic::CosThetaCM(
          6, carbon->GetPDGMass(), 8, oxygen->GetPDGMass(),
          1. * MeV, factor * length);
      Require(std::isfinite(cosine), "Non-finite ZBL scattering cosine");
      Require(cosine >= previous, "ZBL deflection is not impact-monotonic");
      previous = cosine;
    }

    setenv("DNA_ZBL_ALLOW_VALIDATION_PENDING", "1", 1);
    G4DNAZBLFullElastic process(
        carbon, 1. * keV, 100. * MeV, 10. * eV, 10. * eV);
    auto* nist = G4NistManager::Instance();
    auto* water = nist->FindOrBuildMaterial("G4_WATER");
    auto* scaled = nist->BuildMaterialWithNewDensity(
        "ZBL_TEST_ICE", "G4_WATER", 0.94 * g / cm3);
    const G4double waterRate = process.MacroscopicCrossSection(water, 1. * MeV);
    const G4double scaledRate = process.MacroscopicCrossSection(scaled, 1. * MeV);
    Require(waterRate > 0. && scaledRate > 0.,
            "ZBL water macroscopic rate is not positive");
    RequireClose("water density scaling", scaledRate / waterRate, 0.94, 2.e-12);

    const G4double probeEnergy = 1.234 * MeV;
    const G4double directH = G4DNAZBLFullElastic::PairCrossSection(
        6, carbon->GetPDGMass(), 1, G4Proton::ProtonDefinition()->GetPDGMass(),
        probeEnergy, 10. * eV);
    const G4double directO = G4DNAZBLFullElastic::PairCrossSection(
        6, carbon->GetPDGMass(), 8, oxygen->GetPDGMass(),
        probeEnergy, 10. * eV);
    const G4double interpolatedPerMolecule =
        process.MacroscopicCrossSection(water, probeEnergy) /
        G4DNAZBLFullElastic::WaterMoleculeNumberDensity(water);
    RequireClose("off-grid interpolation", interpolatedPerMolecule,
                 2. * directH + directO, 5.e-3);

    std::cout << "Universal-ZBL full elastic validation passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
