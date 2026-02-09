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
/// \file RunAction.cc
/// \brief Implementation of the RunAction class

#include "RunAction.hh"

#include "G4AnalysisManager.hh"
#include "G4Run.hh"
#include "G4Threading.hh"
#include "SteppingAction.hh"
#include "ModelDataRegistry.hh"
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr bool kPrintPostRunStepSummary = false;

bool ReadEnvFlag(const char* name, bool defaultValue)
{
  const char* env = std::getenv(name);
  if (!env || !*env) {
    return defaultValue;
  }
  return std::strcmp(env, "0") != 0;
}

std::string ToLower(std::string value)
{
  for (auto& ch : value) ch = static_cast<char>(std::tolower(ch));
  return value;
}

std::string ReadEnvString(const char* name)
{
  const char* env = std::getenv(name);
  return (env && *env) ? std::string(env) : std::string();
}

std::string NormalizeIcePhase(const std::string& raw)
{
  if (raw.empty()) return {};
  const std::string phase = ToLower(raw);
  if (phase == "ice_hex" || phase == "hex" || phase == "hexagonal" ||
      phase == "crystalline") {
    return "hexagonal";
  }
  if (phase == "ice_am" || phase == "am" || phase == "amo" ||
      phase == "amorphous") {
    return "amorphous";
  }
  return phase;
}

std::string BuildDnaPath(const char* dataDir, const std::string& filename)
{
  if (!dataDir || !*dataDir) {
    return std::string("dna/") + filename;
  }
  return std::string(dataDir) + "/dna/" + filename;
}

bool FileExists(const std::string& path)
{
  std::ifstream test(path.c_str());
  return test.good();
}

std::string SelectIonisationDiffFile(const std::string& icePhase)
{
  const std::string defaultFile = "sigmadiff_ionisation_e_emfietzoglou.dat";
  if (icePhase.empty()) {
    return defaultFile;
  }
  const std::string phaseFile =
      "sigmadiff_ionisation_e_" + icePhase + "_ice_emfietzoglou_kyriakou.dat";
  const char* dataDir = std::getenv("G4LEDATA");
  if (FileExists(BuildDnaPath(dataDir, phaseFile))) {
    return phaseFile;
  }
  return defaultFile;
}

void DeleteOldRootFiles(const std::string& baseName)
{
  namespace fs = std::filesystem;
  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  if (ec) return;
  for (const auto& entry : fs::directory_iterator(cwd, ec)) {
    if (ec || !entry.is_regular_file()) {
      continue;
    }
    const fs::path p = entry.path();
    if (p.extension() != ".root") {
      continue;
    }
    const std::string name = p.filename().string();
    if (name.rfind(baseName, 0) == 0) {
      fs::remove(p, ec);
    }
  }
}
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction() : G4UserRunAction(), fConfigNtupleId(-1)
{
  // Create analysis manager
  G4cout << "##### Create analysis manager "
         << "  " << this << G4endl;
  auto analysisManager = G4AnalysisManager::Instance();

  analysisManager->SetDefaultFileType("root");
  analysisManager->SetNtupleMerging(ReadEnvFlag("DNA_NTUPLE_MERGE", true));

  G4cout << "Using " << analysisManager->GetType() << " analysis manager" << G4endl;

  analysisManager->SetVerboseLevel(1);

  // Creating ntuple

  // Step information ntuple
  analysisManager->CreateNtuple("step", "dnaphysics");
  analysisManager->CreateNtupleDColumn("flagParticle");
  analysisManager->CreateNtupleDColumn("flagProcess");
  analysisManager->CreateNtupleDColumn("x");
  analysisManager->CreateNtupleDColumn("y");
  analysisManager->CreateNtupleDColumn("z");
  analysisManager->CreateNtupleDColumn("totalEnergyDeposit");
  analysisManager->CreateNtupleDColumn("stepLength");
  analysisManager->CreateNtupleDColumn("kineticEnergyDifference");
  analysisManager->CreateNtupleDColumn("kineticEnergy");
  analysisManager->CreateNtupleDColumn("cosTheta");
  analysisManager->CreateNtupleIColumn("eventID");
  analysisManager->CreateNtupleIColumn("trackID");
  analysisManager->CreateNtupleIColumn("parentID");
  analysisManager->CreateNtupleIColumn("stepID");
  // Macroscopic cross-section (1/mm) of the process that defined the step
  analysisManager->CreateNtupleDColumn("vibCrossSection");
  // Optional: channel index for multi-channel processes (e.g., vibrational modes)
  analysisManager->CreateNtupleIColumn("channelIndex");
  // Optional: per-channel microscopic cross-section (cm^2) if available
  analysisManager->CreateNtupleDColumn("channelMicroXS");
  // Optional: process/model name strings
  if (ReadEnvFlag("DNA_NTUPLE_STRINGS", true)) {
    analysisManager->CreateNtupleSColumn("processName");
    analysisManager->CreateNtupleSColumn("modelName");
  }
  analysisManager->FinishNtuple();

  // Track information ntuple
  analysisManager->CreateNtuple("track", "dnaphysics");
  analysisManager->CreateNtupleDColumn("flagParticle");
  analysisManager->CreateNtupleDColumn("x");
  analysisManager->CreateNtupleDColumn("y");
  analysisManager->CreateNtupleDColumn("z");
  analysisManager->CreateNtupleDColumn("dirx");
  analysisManager->CreateNtupleDColumn("diry");
  analysisManager->CreateNtupleDColumn("dirz");
  analysisManager->CreateNtupleDColumn("kineticEnergy");
  analysisManager->CreateNtupleIColumn("trackID");
  analysisManager->CreateNtupleIColumn("parentID");
  analysisManager->FinishNtuple();

  // Configuration metadata ntuple (key/value pairs)
  if (ReadEnvFlag("DNA_NTUPLE_STRINGS", true)) {
    fConfigNtupleId = analysisManager->CreateNtuple("config", "dnaphysics");
    analysisManager->CreateNtupleSColumn("key");
    analysisManager->CreateNtupleSColumn("value");
    analysisManager->FinishNtuple();
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::~RunAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run*)
{
  auto analysisManager = G4AnalysisManager::Instance();

  // Open an output file
  G4String fileName = "dna";
  static G4bool cleaned = false;
  if (!cleaned && G4Threading::IsMasterThread()) {
    if (!ReadEnvFlag("DNA_KEEP_OLD_ROOT", false)) {
      DeleteOldRootFiles(fileName);
    }
    cleaned = true;
  }
  analysisManager->OpenFile(fileName);

  if (fConfigNtupleId >= 0) {
    const std::string physRaw = ReadEnvString("DNA_PHYSICS");
    std::string physChoice = physRaw.empty() ? "ice" : ToLower(physRaw);
    std::string icePhase = NormalizeIcePhase(ReadEnvString("DNA_ICE_PHASE"));

    if (icePhase.empty()) {
      if (physChoice == "ice_hex" || physChoice == "ice_hexagonal" ||
          physChoice == "ice_hexagon" || physChoice == "ice_h") {
        icePhase = "hexagonal";
        physChoice = "ice";
      } else if (physChoice == "ice_am" || physChoice == "ice_amorphous" ||
                 physChoice == "ice_amo") {
        icePhase = "amorphous";
        physChoice = "ice";
      }
    }

    auto appendConfig = [&](const std::string& key, const std::string& value) {
      analysisManager->FillNtupleSColumn(fConfigNtupleId, 0, key);
      analysisManager->FillNtupleSColumn(fConfigNtupleId, 1, value);
      analysisManager->AddNtupleRow(fConfigNtupleId);
    };

    appendConfig("DNA_PHYSICS", physRaw.empty() ? "ice" : physRaw);
    appendConfig("physics_list", physChoice);
    appendConfig("ice_phase", icePhase.empty() ? "default" : icePhase);
  }

  // Clear any previous step logs so this run starts fresh
  SteppingAction::SetLoggingEnabled(kPrintPostRunStepSummary);
  SteppingAction::ClearLogs();
  SteppingAction::ClearObservedModels();
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::EndOfRunAction(const G4Run* aRun)
{
  G4int nofEvents = aRun->GetNumberOfEvent();
  if (nofEvents == 0) return;

  // Print histogram statistics
  auto analysisManager = G4AnalysisManager::Instance();

  if (fConfigNtupleId >= 0) {
    auto observed = SteppingAction::ObservedModels();
    int idx = 0;
    for (const auto& entry : observed) {
      std::ostringstream key;
      key << "observed_model_" << std::setfill('0') << std::setw(3) << idx++;
      analysisManager->FillNtupleSColumn(fConfigNtupleId, 0, key.str());
      analysisManager->FillNtupleSColumn(fConfigNtupleId, 1, entry);
      analysisManager->AddNtupleRow(fConfigNtupleId);
    }

    auto appendConfig = [&](const std::string& key, const std::string& value) {
      analysisManager->FillNtupleSColumn(fConfigNtupleId, 0, key);
      analysisManager->FillNtupleSColumn(fConfigNtupleId, 1, value);
      analysisManager->AddNtupleRow(fConfigNtupleId);
    };

    // Resolve elastic reference based on observed elastic model(s).
    auto refs = ModelDataRegistry::Instance().Snapshot();
    for (const auto& kv : refs) {
      appendConfig(kv.first, kv.second);
    }
  }

  // Save histograms
  analysisManager->Write();
  analysisManager->CloseFile();

  // After the simulation finishes, print custom per-step verbose lines
  if (kPrintPostRunStepSummary) {
    auto& logs = SteppingAction::Logs();
    if (!logs.empty()) {
      G4cout << "\n-- Post-run step summary --" << G4endl;
      G4cout << std::left
             << std::setw(8)  << "Step#"
             << std::setw(16) << "E(eV)"
             << std::setw(28) << "Process"
             << std::setw(28) << "Model"
             << std::setw(16) << "Channel"
             << std::setw(16) << "Sigma(cm^2)"
             << G4endl;

      // numeric formatting
      std::ios::fmtflags oldFlags = G4cout.flags();
      std::streamsize oldPrec = G4cout.precision();
      G4cout.setf(std::ios::scientific);
      G4cout.precision(6);

      for (const auto& r : logs) {
        G4cout << std::right
               << std::setw(8)  << r.stepNo
               << std::setw(16) << r.kinE_eV
               << std::left  << ' ' << std::setw(27) << r.process
               << std::left  << std::setw(27) << (r.model.empty() ? "-" : r.model)
               << std::left  << std::setw(16) << (r.channel.empty() ? "-" : r.channel)
               << std::right << std::setw(16) << r.sigma_area_cm2
               << G4endl;
      }
      // restore
      G4cout.flags(oldFlags);
      G4cout.precision(oldPrec);
    }
  }
}
