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
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publications:
// Med. Phys. 45 (2018) e722-e739
// Phys. Med. 31 (2015) 861-874
// Med. Phys. 37 (2010) 4692-4708
// Int. J. Model. Simul. Sci. Comput. 1 (2010) 157–178
//
// The Geant4-DNA web site is available at http://geant4-dna.org
//
/// \file dnaphysics.cc
/// \brief Implementation of the dnaphysics example

#include "ActionInitialization.hh"
#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "PhysicsList_Water.hh"

#include "G4RunManagerFactory.hh"
#include "G4Types.hh"
#include "G4UIExecutive.hh"
#include "G4VModularPhysicsList.hh"
#include "Randomize.hh"
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <string>
#include "G4UImanager.hh"
#include "G4VisExecutive.hh"

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

int main(int argc, char** argv)
{
  // Detect interactive mode (if no arguments) and define UI session
  G4UIExecutive* ui = nullptr;
  if (argc == 1) {
    ui = new G4UIExecutive(argc, argv);
  }

  // Construct the default run manager
  auto* runManager = G4RunManagerFactory::CreateRunManager();
  if (argc == 3)
    runManager->SetNumberOfThreads(atoi(argv[2]));
  else
    runManager->SetNumberOfThreads(2);

  // Set mandatory user initialization classes
  const char* phys_env = std::getenv("DNA_PHYSICS");
  std::string phys_choice = phys_env ? phys_env : "ice_hex";
  for (auto& c : phys_choice) c = static_cast<char>(std::tolower(c));
  std::string resolved_choice;
  std::string ice_phase;
  if (phys_choice == "water") {
    resolved_choice = "water";
  } else if (phys_choice == "ice_hex") {
    resolved_choice = "ice_hex";
    ice_phase = "hexagonal";
  } else if (phys_choice == "ice_am") {
    resolved_choice = "ice_am";
    ice_phase = "amorphous";
  } else {
    G4cout << "### dnaphysics Warning: unknown DNA_PHYSICS='"
           << phys_choice
           << "'. Supported values: water | ice_hex | ice_am. "
           << "Defaulting to ice_hex." << G4endl;
    resolved_choice = "ice_hex";
    ice_phase = "hexagonal";
  }

  // Canonicalize for all downstream components; phase is now encoded in DNA_PHYSICS.
  setenv("DNA_PHYSICS", resolved_choice.c_str(), 1);

  G4VModularPhysicsList* physlist = nullptr;
  if (resolved_choice == "water") {
    physlist = new PhysicsList_Water();
  } else {
    physlist = new PhysicsList();
  }
  if (!ice_phase.empty()) {
    G4cout << "Using physics list: ice"
           << " (phase: " << ice_phase << ")" << G4endl;
  } else {
    G4cout << "Using physics list: " << resolved_choice << G4endl;
  }
  runManager->SetUserInitialization(new DetectorConstruction(physlist));
  runManager->SetUserInitialization(physlist);

  // User action initialization
  runManager->SetUserInitialization(new ActionInitialization());

  // Visualization
  G4VisExecutive* visManager = nullptr;

  // Get the pointer to the User Interface manager
  G4UImanager* UImanager = G4UImanager::GetUIpointer();

  // Seed RNG so repeated runs are not identical
  // (macro can still override via /random commands if desired)
  CLHEP::HepRandom::setTheSeed(std::time(nullptr));
  if (nullptr == ui) {
    // Batch mode
    G4String command = "/control/execute ";
    G4String fileName = argv[1];
    UImanager->ApplyCommand(command + fileName);
  }
  else {
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
