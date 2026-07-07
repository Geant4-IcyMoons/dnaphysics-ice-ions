#include "PhysicsList_Proton.hh"

#include "G4EmDNAPhysics_option2.hh"
#include "G4EmParameters.hh"
#include "G4PhysicsListHelper.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"

#include "G4DNAIonisation.hh"
#include "G4DNAExcitation.hh"
#include "G4DNAEmfietzoglou_iceProtonIonisationModel.hh"
#include "G4DNAEmfietzoglou_iceProtonExcitationModel.hh"
#include "G4Proton.hh"

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
  const std::string value = ToLower(raw);
  if (value == "1" || value == "true" || value == "on" || value == "yes") {
    return true;
  }
  if (value == "0" || value == "false" || value == "off" || value == "no") {
    return false;
  }
  G4cout << "PhysicsList_Proton: invalid " << key << "='" << raw
         << "', using default " << (defaultValue ? "on" : "off") << G4endl;
  return defaultValue;
}

G4double ReadEnvEnergyEV(const char* key, G4double defaultValue)
{
  const char* raw = std::getenv(key);
  if (!raw || !*raw) return defaultValue;
  char* end = nullptr;
  const G4double value = std::strtod(raw, &end);
  if (end != raw && value > 0.) return value * eV;
  G4cout << "PhysicsList_Proton: invalid " << key << "='" << raw
         << "', using default " << defaultValue / eV << " eV" << G4endl;
  return defaultValue;
}
}  // namespace

PhysicsList_Proton::PhysicsList_Proton() : G4VModularPhysicsList()
{
  SetDefaultCutValue(1.0 * micrometer);
  SetVerboseLevel(1);

  // Keep Geant4-DNA particle definitions available.
  fEmPhysicsList = new G4EmDNAPhysics_option2();

  G4ProductionCutsTable::GetProductionCutsTable()->SetEnergyRange(10.0 * eV, 1.0 * GeV);
  auto* param = G4EmParameters::Instance();
  param->SetMinEnergy(10.0 * eV);
  param->SetMaxEnergy(1.0 * GeV);
}

PhysicsList_Proton::~PhysicsList_Proton() = default;

void PhysicsList_Proton::ConstructParticle()
{
  fEmPhysicsList->ConstructParticle();
}

void PhysicsList_Proton::ConstructProcess()
{
  AddTransportation();

  auto* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  auto* proton = G4Proton::ProtonDefinition();

  const G4bool enableExcitation =
      ReadEnvFlag("DNA_PROTON_ENABLE_EXCITATION", true);
  const G4bool enableIonisation =
      ReadEnvFlag("DNA_PROTON_ENABLE_IONISATION", true);
  const G4bool useBarkas =
      ReadEnvFlag("DNA_PROTON_BARKAS_DCS",
                  ReadEnvFlag("DNA_ICE_PROTON_BARKAS_DCS", false));
  G4double protonMin = ReadEnvEnergyEV("DNA_PROTON_MIN_ENERGY_EV", 1.0e5 * eV);
  G4double protonMax = ReadEnvEnergyEV("DNA_PROTON_MAX_ENERGY_EV", 1.0e8 * eV);
  if (protonMin >= protonMax) {
    G4cout << "PhysicsList_Proton: invalid proton energy window, using "
           << "0.1-100 MeV" << G4endl;
    protonMin = 0.1 * MeV;
    protonMax = 100. * MeV;
  }

  G4cout << "PhysicsList_Proton: proton-only ice DCS models"
         << " [excitation=" << (enableExcitation ? "on" : "off")
         << ", ionisation=" << (enableIonisation ? "on" : "off")
         << ", barkas_dcs=" << (useBarkas ? "on" : "off")
         << ", range=" << protonMin / MeV << "-" << protonMax / MeV
         << " MeV]" << G4endl;

  if (ReadEnvFlag("DNA_PROTON_ENABLE_CHARGE_EXCHANGE", false)) {
    G4cout << "PhysicsList_Proton: charge exchange is not registered in this "
           << "proton-only DCS pass; leaving it off." << G4endl;
  }

  if (enableExcitation) {
    auto* protonExcitation = new G4DNAExcitation("proton_G4DNAExcitation");
    auto* protonExcitationModel = new G4DNAEmfietzoglou_iceProtonExcitationModel();
    protonExcitationModel->SelectStationary(false);
    protonExcitationModel->SetLowEnergyLimit(protonMin);
    protonExcitationModel->SetHighEnergyLimit(protonMax);
    protonExcitation->SetEmModel(protonExcitationModel);
    protonExcitation->SetMinKinEnergy(protonMin);
    protonExcitation->SetMaxKinEnergy(protonMax);
    ph->RegisterProcess(protonExcitation, proton);
  }

  if (enableIonisation) {
    auto* protonIonisation = new G4DNAIonisation("proton_G4DNAIonisation");
    auto* protonIonisationModel = new G4DNAEmfietzoglou_iceProtonIonisationModel();
    protonIonisationModel->SelectStationary(false);
    protonIonisationModel->SetLowEnergyLimit(protonMin);
    protonIonisationModel->SetHighEnergyLimit(protonMax);
    protonIonisation->SetEmModel(protonIonisationModel);
    protonIonisation->SetMinKinEnergy(protonMin);
    protonIonisation->SetMaxKinEnergy(protonMax);
    ph->RegisterProcess(protonIonisation, proton);
  }
}

void PhysicsList_Proton::AddPhysics(const G4String&)
{
}

void PhysicsList_Proton::TrackingCut()
{
}

void PhysicsList_Proton::SetTrackingCut(G4bool)
{
}
