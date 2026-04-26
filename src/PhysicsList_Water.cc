//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of the scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
/// \file PhysicsList_Water.cc
/// \brief Explicit G4-DNA water physics list with per-process model toggles.

#include "PhysicsList_Water.hh"

#include "G4CoulombScattering.hh"
#include "G4DNAElastic.hh"
#include "G4DNAElectronSolvation.hh"
#include "G4DNAEmfietzoglouExcitationModel.hh"
#include "G4DNAEmfietzoglouIonisationModel.hh"
#include "G4DNAExcitation.hh"
#include "G4DNAIonisation.hh"
#include "G4DNAOneStepThermalizationModel.hh"
#include "G4DNABornExcitationModel.hh"
#include "G4DNABornIonisationModel.hh"
#include "G4DNAUeharaScreenedRutherfordElasticModel.hh"
#include "G4Electron.hh"
#include "G4EmDNABuilder.hh"
#include "G4EmParameters.hh"
#include "G4MollerBhabhaModel.hh"
#include "G4PhysicsListHelper.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SeltzerBergerModel.hh"
#include "G4SystemOfUnits.hh"
#include "G4UrbanMscModel.hh"
#include "G4WentzelVIModel.hh"
#include "G4eBremsstrahlung.hh"
#include "G4eIonisation.hh"
#include "G4eMultipleScattering.hh"

#include <cctype>
#include <cstdlib>
#include <string>

namespace {
std::string ToLower(std::string value)
{
  for (auto& ch : value) ch = static_cast<char>(std::tolower(ch));
  return value;
}

G4bool ReadEnvFlag(const char* key, G4bool defaultValue)
{
  const char* raw = std::getenv(key);
  if (!raw || !*raw) return defaultValue;
  const std::string v = ToLower(raw);
  if (v == "1" || v == "true" || v == "on" || v == "yes") return true;
  if (v == "0" || v == "false" || v == "off" || v == "no") return false;
  G4cout << "PhysicsList_Water: invalid " << key << "='" << raw
         << "', using default " << (defaultValue ? "on" : "off") << G4endl;
  return defaultValue;
}
}  // namespace

PhysicsList_Water::PhysicsList_Water() : G4VModularPhysicsList()
{
  SetDefaultCutValue(1.0 * micrometer);
  SetVerboseLevel(1);

  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(1 * eV, 1 * GeV);
  G4EmParameters* param = G4EmParameters::Instance();
  param->SetMinEnergy(1 * eV);
  param->SetMaxEnergy(1 * GeV);
}

PhysicsList_Water::~PhysicsList_Water()
{
  // No owned physics constructors.
}

void PhysicsList_Water::ConstructParticle()
{
  // Build all DNA particles used by the explicit process registration below.
  G4EmDNABuilder::ConstructDNAParticles();
}

void PhysicsList_Water::ConstructProcess()
{
  AddTransportation();

  // Runtime toggles:
  //   DNA_WATER_ENABLE_ELASTIC           (default on)
  //   DNA_WATER_ENABLE_EXCITATION        (default on)
  //   DNA_WATER_ENABLE_EXCITATION_EMF    (default on)
  //   DNA_WATER_ENABLE_EXCITATION_BORN   (default on)
  //   DNA_WATER_ENABLE_IONISATION        (default on)
  //   DNA_WATER_ENABLE_IONISATION_EMF    (default on)
  //   DNA_WATER_ENABLE_IONISATION_BORN   (default on)
  //   DNA_WATER_ENABLE_SOLVATION         (default on)
  //   DNA_WATER_ENABLE_HIGH_EM           (default on)
  const G4bool enableElastic = ReadEnvFlag("DNA_WATER_ENABLE_ELASTIC", true);
  const G4bool enableExcitation = ReadEnvFlag("DNA_WATER_ENABLE_EXCITATION", true);
  const G4bool enableExcitationEmf = ReadEnvFlag("DNA_WATER_ENABLE_EXCITATION_EMF", true);
  const G4bool enableExcitationBorn = ReadEnvFlag("DNA_WATER_ENABLE_EXCITATION_BORN", true);
  const G4bool enableIonisation = ReadEnvFlag("DNA_WATER_ENABLE_IONISATION", true);
  const G4bool enableIonisationEmf = ReadEnvFlag("DNA_WATER_ENABLE_IONISATION_EMF", true);
  const G4bool enableIonisationBorn = ReadEnvFlag("DNA_WATER_ENABLE_IONISATION_BORN", true);
  const G4bool enableSolvation = ReadEnvFlag("DNA_WATER_ENABLE_SOLVATION", true);
  const G4bool enableHighEM = ReadEnvFlag("DNA_WATER_ENABLE_HIGH_EM", true);

  G4cout << "PhysicsList_Water: explicit model registration"
         << " [elastic=" << (enableElastic ? "on" : "off")
         << ", excitation=" << (enableExcitation ? "on" : "off")
         << " (emf=" << (enableExcitationEmf ? "on" : "off")
         << ", born=" << (enableExcitationBorn ? "on" : "off") << ")"
         << ", ionisation=" << (enableIonisation ? "on" : "off")
         << " (emf=" << (enableIonisationEmf ? "on" : "off")
         << ", born=" << (enableIonisationBorn ? "on" : "off") << ")"
         << ", solvation=" << (enableSolvation ? "on" : "off")
         << ", highEM=" << (enableHighEM ? "on" : "off") << "]"
         << G4endl;

  auto* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  auto* electron = G4Electron::ElectronDefinition();

  if (enableElastic) {
    auto* process = new G4DNAElastic("e-_G4DNAElastic");
    // water option4 electron elastic branch
    auto* model = new G4DNAUeharaScreenedRutherfordElasticModel();
    model->SetLowEnergyLimit(100. * eV);
    model->SetHighEnergyLimit(1. * MeV);
    process->SetEmModel(model);
    process->SetMinKinEnergy(100. * eV);
    process->SetMaxKinEnergy(1. * MeV);
    ph->RegisterProcess(process, electron);
  }

  if (enableExcitation) {
    auto* process = new G4DNAExcitation("e-_G4DNAExcitation");
    G4bool hasModel = false;
    if (enableExcitationEmf) {
      auto* model = new G4DNAEmfietzoglouExcitationModel();
      model->SetLowEnergyLimit(8. * eV);
      model->SetHighEnergyLimit(10. * keV);
      process->SetEmModel(model);
      hasModel = true;
    }
    if (enableExcitationBorn) {
      auto* model = new G4DNABornExcitationModel();
      model->SetLowEnergyLimit(10. * keV);
      model->SetHighEnergyLimit(1. * MeV);
      if (!hasModel) {
        process->SetEmModel(model);
      } else {
        process->AddEmModel(2, model);
      }
      hasModel = true;
    }
    if (hasModel) {
      process->SetMinKinEnergy(8. * eV);
      process->SetMaxKinEnergy(1. * MeV);
      ph->RegisterProcess(process, electron);
    } else {
      delete process;
    }
  }

  G4DNAIonisation* dnaIonisationProcess = nullptr;
  if (enableIonisation) {
    auto* process = new G4DNAIonisation("e-_G4DNAIonisation");
    G4bool hasModel = false;
    if (enableIonisationEmf) {
      auto* model = new G4DNAEmfietzoglouIonisationModel();
      model->SetLowEnergyLimit(10. * eV);
      model->SetHighEnergyLimit(10. * keV);
      process->SetEmModel(model);
      hasModel = true;
    }
    if (enableIonisationBorn) {
      auto* model = new G4DNABornIonisationModel();
      model->SetLowEnergyLimit(10. * keV);
      model->SetHighEnergyLimit(1. * MeV);
      if (!hasModel) {
        process->SetEmModel(model);
      } else {
        process->AddEmModel(2, model);
      }
      hasModel = true;
    }
    if (hasModel) {
      process->SetMinKinEnergy(10. * eV);
      process->SetMaxKinEnergy(1. * GeV);
      ph->RegisterProcess(process, electron);
      dnaIonisationProcess = process;
    } else {
      delete process;
    }
  }

  if (enableSolvation) {
    auto* process = new G4DNAElectronSolvation("e-_G4DNAElectronSolvation");
    auto* model = new G4DNAOneStepThermalizationModel();
    model->SetLowEnergyLimit(1. * eV);
    model->SetHighEnergyLimit(10. * eV);
    process->SetEmModel(model);
    process->SetMinKinEnergy(1. * eV);
    process->SetMaxKinEnergy(10. * eV);
    ph->RegisterProcess(process, electron);
  }

  if (enableHighEM) {
    // Keep a high-energy fallback branch so disabling DNA components
    // does not leave electrons without transport above 1 MeV.
    const G4double kHighMin = 1. * MeV;
    const G4double kHighMax = 1. * GeV;

    auto* msc = new G4eMultipleScattering();
    auto* mscModel = new G4UrbanMscModel();
    mscModel->SetLowEnergyLimit(kHighMin);
    mscModel->SetHighEnergyLimit(kHighMax);
    msc->SetEmModel(mscModel);
    ph->RegisterProcess(msc, electron);

    auto* highIonModel = new G4MollerBhabhaModel();
    highIonModel->SetLowEnergyLimit(kHighMin);
    highIonModel->SetHighEnergyLimit(kHighMax);
    if (dnaIonisationProcess != nullptr) {
      dnaIonisationProcess->AddEmModel(10, highIonModel);
    } else {
      auto* ionisation = new G4eIonisation();
      ionisation->SetEmModel(highIonModel);
      ph->RegisterProcess(ionisation, electron);
    }

    auto* brem = new G4eBremsstrahlung();
    brem->SetMinKinEnergy(kHighMin);
    auto* bremModel = new G4SeltzerBergerModel();
    bremModel->SetLowEnergyLimit(kHighMin);
    bremModel->SetHighEnergyLimit(kHighMax);
    brem->SetEmModel(bremModel);
    ph->RegisterProcess(brem, electron);

    auto* cs = new G4CoulombScattering();
    cs->SetMinKinEnergy(kHighMin);
    auto* csModel = new G4WentzelVIModel();
    csModel->SetLowEnergyLimit(kHighMin);
    csModel->SetHighEnergyLimit(kHighMax);
    cs->SetEmModel(csModel);
    ph->RegisterProcess(cs, electron);
  }
}
