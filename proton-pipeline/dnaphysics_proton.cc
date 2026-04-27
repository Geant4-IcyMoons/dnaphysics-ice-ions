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

#include <cctype>
#include <cstdlib>
#include <ctime>
#include <string>

int main(int argc, char** argv)
{
  G4UIExecutive* ui = nullptr;
  if (argc == 1) {
    ui = new G4UIExecutive(argc, argv);
  }

  auto* runManager = G4RunManagerFactory::CreateRunManager();
  if (argc == 3) {
    runManager->SetNumberOfThreads(std::atoi(argv[2]));
  } else {
    runManager->SetNumberOfThreads(2);
  }

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

  G4cout << "Using proton physics list (DNA_PHYSICS=" << phys_choice << ")" << G4endl;

  auto* physlist = new PhysicsList_Proton();
  runManager->SetUserInitialization(new DetectorConstruction(physlist));
  runManager->SetUserInitialization(physlist);
  runManager->SetUserInitialization(new ActionInitialization());

  G4VisExecutive* visManager = nullptr;
  G4UImanager* UImanager = G4UImanager::GetUIpointer();

  CLHEP::HepRandom::setTheSeed(std::time(nullptr));

  if (nullptr == ui) {
    G4String command = "/control/execute ";
    G4String fileName = argv[1];
    UImanager->ApplyCommand(command + fileName);
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
