#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList_Proton.hh"

#include "G4RunManagerFactory.hh"
#include "G4Types.hh"
#include "G4UIExecutive.hh"
#include "G4UImanager.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VisExecutive.hh"
#include "Randomize.hh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <string>

namespace
{
std::string ReadEnvString(const char* key, const char* legacyKey, const std::string& defaultValue)
{
  const char* raw = std::getenv(key);
  if (raw && *raw) return raw;
  raw = legacyKey ? std::getenv(legacyKey) : nullptr;
  if (raw && *raw) return raw;
  return defaultValue;
}

G4double ParsePositiveDouble(const std::string& text, G4double fallback, const char* label)
{
  char* end = nullptr;
  const G4double value = std::strtod(text.c_str(), &end);
  if (end != text.c_str() && value > 0.) return value;
  G4cout << "### dnaphysics_proton Warning: invalid " << label << "='" << text
         << "', using " << fallback << G4endl;
  return fallback;
}

long ParsePositiveLong(const std::string& text, long fallback, const char* label)
{
  char* end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (end != text.c_str() && *end == '\0' && value > 0) return value;
  G4cout << "### dnaphysics_proton Warning: invalid " << label << "='" << text
         << "', using " << fallback << G4endl;
  return fallback;
}

void ApplySourceAndRun(G4UImanager* ui,
                       const std::string& particle,
                       G4double eminMeV,
                       G4double emaxMeV,
                       const std::string& events,
                       const std::string& number)
{
  if (emaxMeV < eminMeV) {
    G4cout << "### dnaphysics_proton Warning: Emax < Emin; swapping source energy bounds."
           << G4endl;
    const G4double tmp = eminMeV;
    eminMeV = emaxMeV;
    emaxMeV = tmp;
  }

  if (particle == "carbon" || particle == "oxygen" || particle == "sulfur") {
    G4int z = 6;
    G4int a = 12;
    if (particle == "oxygen") {
      z = 8;
      a = 16;
    } else if (particle == "sulfur") {
      z = 16;
      a = 32;
    }
    ui->ApplyCommand("/gps/particle ion");
    // Fully stripped ions are the source default. ZBL and NLH use nuclear Z,
    // not this ionic charge state.
    ui->ApplyCommand("/gps/ion " + G4String(std::to_string(z)) + " " +
                     G4String(std::to_string(a)) + " " +
                     G4String(std::to_string(z)));
  } else {
    ui->ApplyCommand("/gps/particle " + G4String(particle));
  }
  ui->ApplyCommand("/gps/number " + G4String(number));
  ui->ApplyCommand("/gps/pos/type Point");
  ui->ApplyCommand("/gps/pos/centre 0 0 -0.01 mm");
  ui->ApplyCommand("/gps/ang/type beam1d");
  ui->ApplyCommand("/gps/direction 0 0 1");

  if (std::fabs(emaxMeV - eminMeV) <= 1.0e-12 * std::max(1.0, std::fabs(eminMeV))) {
    ui->ApplyCommand("/gps/ene/type Mono");
    ui->ApplyCommand("/gps/ene/mono " + G4String(std::to_string(eminMeV)) + " MeV");
    G4cout << "Source energy mode: mono " << eminMeV << " MeV" << G4endl;
  } else {
    ui->ApplyCommand("/gps/ene/type Lin");
    ui->ApplyCommand("/gps/ene/min " + G4String(std::to_string(eminMeV)) + " MeV");
    ui->ApplyCommand("/gps/ene/max " + G4String(std::to_string(emaxMeV)) + " MeV");
    ui->ApplyCommand("/gps/ene/gradient 0.");
    ui->ApplyCommand("/gps/ene/intercept 1.");
    G4cout << "Source energy mode: uniform " << eminMeV << "-" << emaxMeV
           << " MeV" << G4endl;
  }

  ui->ApplyCommand("/run/beamOn " + G4String(events));
}
}  // namespace

int main(int argc, char** argv)
{
  G4UIExecutive* ui = nullptr;
  if (argc == 1) {
    ui = new G4UIExecutive(argc, argv);
  }

  auto* runManager = G4RunManagerFactory::CreateRunManager();
  if (argc >= 3) {
    runManager->SetNumberOfThreads(std::atoi(argv[2]));
  } else {
    runManager->SetNumberOfThreads(2);
  }

  std::string sourceParticle =
    ReadEnvString("DNA_SOURCE_PARTICLE", "DNA_PROTON_SOURCE_PARTICLE", "proton");
  const std::string defaultSourceEnergyMeV =
    ReadEnvString("DNA_SOURCE_ENERGY_MEV", "DNA_PROTON_SOURCE_ENERGY_MEV", "0.5");
  std::string sourceEminMeV =
    ReadEnvString("DNA_SOURCE_EMIN_MEV", "DNA_PROTON_SOURCE_EMIN_MEV", defaultSourceEnergyMeV);
  std::string sourceEmaxMeV =
    ReadEnvString("DNA_SOURCE_EMAX_MEV", "DNA_PROTON_SOURCE_EMAX_MEV", defaultSourceEnergyMeV);
  std::string sourceEvents =
    ReadEnvString("DNA_SOURCE_EVENTS", "DNA_PROTON_SOURCE_EVENTS", "10");
  std::string sourceNumber =
    ReadEnvString("DNA_SOURCE_NUMBER", "DNA_PROTON_SOURCE_NUMBER", "1");

  // CLI form:
  //   dnaphysics_proton macro.mac threads particle Emin_MeV Emax_MeV events particles_per_event
  if (argc >= 4) sourceParticle = argv[3];
  if (argc >= 5) sourceEminMeV = argv[4];
  if (argc >= 6) sourceEmaxMeV = argv[5];
  if (argc >= 7) sourceEvents = argv[6];
  if (argc >= 8) sourceNumber = argv[7];

  for (auto& c : sourceParticle) c = static_cast<char>(std::tolower(c));
  if (sourceParticle != "proton" && sourceParticle != "alpha" &&
      sourceParticle != "carbon" && sourceParticle != "oxygen" &&
      sourceParticle != "sulfur") {
    G4cerr << "### dnaphysics_proton Error: unsupported source particle '"
           << sourceParticle
           << "'. Use proton, alpha, carbon, oxygen, or sulfur." << G4endl;
    delete runManager;
    return 2;
  }

  const G4double sourceEminValueMeV =
    ParsePositiveDouble(sourceEminMeV, 0.5, "source Emin_MeV");
  const G4double sourceEmaxValueMeV =
    ParsePositiveDouble(sourceEmaxMeV, sourceEminValueMeV, "source Emax_MeV");

  // Keep phase selection compatible with existing data naming.
  const char* phys_env = std::getenv("DNA_PHYSICS");
  std::string phys_choice = phys_env ? phys_env : "ice_hex";
  for (auto& c : phys_choice) c = static_cast<char>(std::tolower(c));

  if (phys_choice != "water" && phys_choice != "ice_hex" && phys_choice != "ice_am") {
    G4cout << "### dnaphysics_proton Warning: unknown DNA_PHYSICS='"
           << phys_choice
           << "'. Supported values: water | ice_hex | ice_am. "
           << "Defaulting to ice_hex." << G4endl;
    phys_choice = "ice_hex";
  }
  setenv("DNA_PHYSICS", phys_choice.c_str(), 1);
  setenv("DNA_SOURCE_PARTICLE", sourceParticle.c_str(), 1);
  setenv("DNA_SOURCE_EMIN_MEV", sourceEminMeV.c_str(), 1);
  setenv("DNA_SOURCE_EMAX_MEV", sourceEmaxMeV.c_str(), 1);
  setenv("DNA_SOURCE_EVENTS", sourceEvents.c_str(), 1);
  setenv("DNA_SOURCE_NUMBER", sourceNumber.c_str(), 1);
  if (sourceParticle == "carbon" && !std::getenv("DNA_ION_ELASTIC_MODEL") &&
      !std::getenv("DNA_ION_HARD_ELASTIC")) {
    setenv("DNA_ION_HARD_ELASTIC", "1", 1);
  }

  G4cout << "Using H/He/C/O/S ion physics list (DNA_PHYSICS="
         << phys_choice << ")" << G4endl;
  G4cout << "Source settings: particle=" << sourceParticle
         << ", energy=" << sourceEminValueMeV << "-" << sourceEmaxValueMeV << " MeV"
         << ", events=" << sourceEvents
         << ", gps_number=" << sourceNumber << G4endl;

  auto* physlist = new PhysicsList_Proton();
  runManager->SetUserInitialization(new DetectorConstruction(physlist));
  runManager->SetUserInitialization(physlist);
  runManager->SetUserInitialization(new ActionInitialization());

  G4VisExecutive* visManager = nullptr;
  G4UImanager* UImanager = G4UImanager::GetUIpointer();

  const long fallbackSeed = static_cast<long>(std::time(nullptr));
  const long randomSeed = ParsePositiveLong(
      ReadEnvString("DNA_RANDOM_SEED", nullptr, std::to_string(fallbackSeed)),
      fallbackSeed,
      "DNA_RANDOM_SEED");
  CLHEP::HepRandom::setTheSeed(randomSeed);
  setenv("DNA_RANDOM_SEED", std::to_string(randomSeed).c_str(), 1);
  G4cout << "Random seed: " << randomSeed << G4endl;

  if (nullptr == ui) {
    G4String command = "/control/execute ";
    G4String fileName = argv[1];
    UImanager->ApplyCommand(command + fileName);
    ApplySourceAndRun(UImanager,
                      sourceParticle,
                      sourceEminValueMeV,
                      sourceEmaxValueMeV,
                      sourceEvents,
                      sourceNumber);
  } else {
    visManager = new G4VisExecutive;
    visManager->Initialize();
    UImanager->ApplyCommand("/control/execute vis.mac");
    ui->SessionStart();
    delete ui;
    delete visManager;
  }

  delete runManager;
  return 0;
}
