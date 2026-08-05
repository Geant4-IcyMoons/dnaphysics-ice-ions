#include "PhysicsList_Proton.hh"

#include "G4EmDNAPhysics_option2.hh"
#include "G4EmDNABuilder.hh"
#include "G4EmParameters.hh"
#include "G4Exception.hh"
#include "G4PhysicsListHelper.hh"
#include "G4ProductionCutsTable.hh"
#include "G4SystemOfUnits.hh"

#include "G4Alpha.hh"
#include "G4DNAChargeDecrease.hh"
#include "G4DNADingfelderChargeDecreaseModel.hh"
#include "G4DNANLHHardElastic.hh"
#include "G4DNAGenericIonsManager.hh"
#include "G4DNAIonisation.hh"
#include "G4DNAExcitation.hh"
#include "G4DNAEmfietzoglou_iceProtonIonisationModel.hh"
#include "G4DNAEmfietzoglou_iceProtonExcitationModel.hh"
#include "G4Proton.hh"
#include "G4IonConstructor.hh"
#include "G4IonTable.hh"

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

G4bool HasEnv(const char* key)
{
  const char* raw = std::getenv(key);
  return raw && *raw;
}

std::string ReadEnvString(const char* key, const std::string& defaultValue)
{
  const char* raw = std::getenv(key);
  return (raw && *raw) ? std::string(raw) : defaultValue;
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
  G4IonConstructor ions;
  ions.ConstructParticle();
}

void PhysicsList_Proton::ConstructProcess()
{
  AddTransportation();

  auto* ph = G4PhysicsListHelper::GetPhysicsListHelper();
  const std::string sourceParticle =
      ToLower(ReadEnvString("DNA_SOURCE_PARTICLE", "proton"));
  G4ParticleDefinition* projectile = nullptr;
  if (sourceParticle == "proton") {
    projectile = G4Proton::ProtonDefinition();
  } else if (sourceParticle == "alpha") {
    projectile = G4Alpha::AlphaDefinition();
  } else if (sourceParticle == "carbon") {
    projectile = G4IonTable::GetIonTable()->GetIon(6, 12, 0.);
  } else {
    G4ExceptionDescription description;
    description << "Unsupported source particle '" << sourceParticle
                << "'. This pipeline supports proton, alpha, or carbon-12.";
    G4Exception("PhysicsList_Proton::ConstructProcess", "ionphysics001",
                FatalException, description);
    return;
  }

  const G4bool enableExcitation =
      HasEnv("DNA_ION_ENABLE_EXCITATION")
          ? ReadEnvFlag("DNA_ION_ENABLE_EXCITATION", sourceParticle != "carbon")
          : ReadEnvFlag("DNA_PROTON_ENABLE_EXCITATION", sourceParticle != "carbon");
  const G4bool enableIonisation =
      HasEnv("DNA_ION_ENABLE_IONISATION")
          ? ReadEnvFlag("DNA_ION_ENABLE_IONISATION", sourceParticle != "carbon")
          : ReadEnvFlag("DNA_PROTON_ENABLE_IONISATION", sourceParticle != "carbon");
  const G4bool useBarkas =
      HasEnv("DNA_ION_BARKAS_DCS")
          ? ReadEnvFlag("DNA_ION_BARKAS_DCS", false)
          : ReadEnvFlag("DNA_PROTON_BARKAS_DCS",
                        ReadEnvFlag("DNA_ICE_PROTON_BARKAS_DCS", false));
  const G4bool enableChargeExchange =
      HasEnv("DNA_ION_CHARGE_EXCHANGE")
          ? ReadEnvFlag("DNA_ION_CHARGE_EXCHANGE", false)
          : ReadEnvFlag("DNA_PROTON_ENABLE_CHARGE_EXCHANGE", false);
  const G4bool enableHardElastic = ReadEnvFlag(
      "DNA_ION_HARD_ELASTIC", sourceParticle == "carbon");
  G4double ionMin =
      HasEnv("DNA_ION_MIN_ENERGY_EV")
          ? ReadEnvEnergyEV("DNA_ION_MIN_ENERGY_EV",
                            (sourceParticle == "carbon" ? 1.0e3 : 1.0e5) * eV)
          : ReadEnvEnergyEV("DNA_PROTON_MIN_ENERGY_EV",
                            (sourceParticle == "carbon" ? 1.0e3 : 1.0e5) * eV);
  G4double ionMax =
      HasEnv("DNA_ION_MAX_ENERGY_EV")
          ? ReadEnvEnergyEV("DNA_ION_MAX_ENERGY_EV", 1.0e8 * eV)
          : ReadEnvEnergyEV("DNA_PROTON_MAX_ENERGY_EV", 1.0e8 * eV);
  if (ionMin >= ionMax) {
    G4cout << "PhysicsList_Proton: invalid ion energy window, using "
           << "0.1-100 MeV" << G4endl;
    ionMin = 0.1 * MeV;
    ionMax = 100. * MeV;
  }

  if (sourceParticle == "carbon" && (enableExcitation || enableIonisation)) {
    G4Exception("PhysicsList_Proton::ConstructProcess", "ionphysics002",
                FatalException,
                "The existing electronic excitation/ionisation models are "
                "validated only for proton and alpha. Disable them for the "
                "carbon hard-elastic run.");
  }
  if (sourceParticle == "carbon" && enableChargeExchange) {
    G4Exception("PhysicsList_Proton::ConstructProcess", "ionphysics003",
                FatalException,
                "Dingfelder charge exchange does not support carbon. The "
                "carbon charge-state process must be registered separately.");
  }
  if (sourceParticle == "carbon" && useBarkas) {
    G4Exception("PhysicsList_Proton::ConstructProcess", "ionphysics004",
                FatalException,
                "DNA_ION_BARKAS_DCS selects proton/alpha electronic DCS "
                "tables and is not part of the carbon NLH hard-elastic "
                "model. Disable it for this carbon-only configuration.");
  }
  if (sourceParticle != "carbon" && enableHardElastic) {
    G4Exception("PhysicsList_Proton::ConstructProcess", "ionphysics005",
                FatalException,
                "The released NLH Geant4 model is carbon-only. HTran is a "
                "complete H/He elastic treatment; NLH cannot overlap it without "
                "a validated handoff or angular/impact-parameter partition.");
  }

  G4cout << "PhysicsList_Proton: " << sourceParticle << " ice DCS models"
         << " [excitation=" << (enableExcitation ? "on" : "off")
         << ", ionisation=" << (enableIonisation ? "on" : "off")
         << ", barkas_dcs=" << (useBarkas ? "on" : "off")
         << ", charge_exchange=" << (enableChargeExchange ? "on" : "off")
         << ", nlh_hard_elastic=" << (enableHardElastic ? "on" : "off")
         << ", range=" << ionMin / MeV << "-" << ionMax / MeV
         << " MeV]" << G4endl;

  if (enableHardElastic) {
    G4cout << "PhysicsList_Proton: registering threshold-defined NLH hard "
           << "elastic for " << projectile->GetParticleName()
           << " [Z=" << projectile->GetAtomicNumber()
           << ", A=" << projectile->GetAtomicMass() << "]." << G4endl;
    if (!ph->RegisterProcess(new G4DNANLHHardElastic(), projectile)) {
      G4Exception("PhysicsList_Proton::ConstructProcess", "ionphysics006",
                  FatalException,
                  "Could not register the carbon NLH hard-elastic process.");
    }
  }

  if (enableExcitation) {
    auto* excitation = new G4DNAExcitation(
        G4String(sourceParticle) + "_G4DNAExcitation");
    auto* excitationModel = new G4DNAEmfietzoglou_iceProtonExcitationModel(
        projectile,
        G4String("DNAEmfietzoglou_ice") +
            (sourceParticle == "proton" ? "Proton" : "Alpha") +
            "ExcitationModel");
    excitationModel->SelectStationary(false);
    excitationModel->SetLowEnergyLimit(ionMin);
    excitationModel->SetHighEnergyLimit(ionMax);
    excitation->SetEmModel(excitationModel);
    excitation->SetMinKinEnergy(ionMin);
    excitation->SetMaxKinEnergy(ionMax);
    ph->RegisterProcess(excitation, projectile);
  }

  if (enableIonisation) {
    auto* ionisation = new G4DNAIonisation(
        G4String(sourceParticle) + "_G4DNAIonisation");
    auto* ionisationModel = new G4DNAEmfietzoglou_iceProtonIonisationModel(
        projectile,
        G4String("DNAEmfietzoglou_ice") +
            (sourceParticle == "proton" ? "Proton" : "Alpha") +
            "IonisationModel");
    ionisationModel->SelectStationary(false);
    ionisationModel->SetLowEnergyLimit(ionMin);
    ionisationModel->SetHighEnergyLimit(ionMax);
    ionisation->SetEmModel(ionisationModel);
    ionisation->SetMinKinEnergy(ionMin);
    ionisation->SetMaxKinEnergy(ionMax);
    ph->RegisterProcess(ionisation, projectile);
  }

  if (enableChargeExchange) {
    auto* chargeDecrease = G4EmDNABuilder::FindOrBuildChargeDecrease(
        projectile, G4String(sourceParticle) + "_G4DNAChargeDecrease");
    auto* decreaseModel = new G4DNADingfelderChargeDecreaseModel();
    decreaseModel->SelectStationary(false);
    decreaseModel->SetLowEnergyLimit(0.0);
    decreaseModel->SetHighEnergyLimit(
        sourceParticle == "proton" ? 100. * MeV : 400. * MeV);
    chargeDecrease->AddEmModel(-1, decreaseModel);

    auto* ionManager = G4DNAGenericIonsManager::Instance();
    if (sourceParticle == "proton") {
      auto* hydrogen = ionManager->GetIon("hydrogen");
      G4EmDNABuilder::ConstructDNALightIonPhysics(
          hydrogen, 0, 2, 100. * MeV, true, false);
      G4cout << "PhysicsList_Proton: Dingfelder charge exchange registered "
             << "for proton <-> hydrogen." << G4endl;
    } else {
      auto* alphaPlus = ionManager->GetIon("alpha+");
      auto* helium = ionManager->GetIon("helium");
      G4EmDNABuilder::ConstructDNALightIonPhysics(
          alphaPlus, 1, 2, 400. * MeV, true, false);
      G4EmDNABuilder::ConstructDNALightIonPhysics(
          helium, 0, 2, 400. * MeV, true, false);
      G4cout << "PhysicsList_Proton: Dingfelder charge exchange registered "
             << "for alpha++ <-> alpha+ <-> helium." << G4endl;
    }
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
