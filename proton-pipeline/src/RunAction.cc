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

#include "G4AccumulableManager.hh"
#include "G4AnalysisManager.hh"
#include "G4Exception.hh"
#include "G4Run.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4UnitsTable.hh"
#include "G4ios.hh"
#include "SteppingAction.hh"
#include "ModelDataRegistry.hh"
#include <cmath>
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

long ReadEnvLong(const char* name, long defaultValue)
{
  const char* env = std::getenv(name);
  if (!env || !*env) {
    return defaultValue;
  }
  char* end = nullptr;
  const long val = std::strtol(env, &end, 10);
  if (end == env) {
    return defaultValue;
  }
  return val;
}

bool HasEnv(const char* name)
{
  const char* env = std::getenv(name);
  return (env && *env);
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
  if (phase == "water") return {};
  if (phase == "ice_hex") {
    return "hexagonal";
  }
  if (phase == "ice_am") {
    return "amorphous";
  }
  return {};
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

RunAction::LogMode RunAction::ParseLogMode(const std::string& mode)
{
  const std::string key = ToLower(mode);
  if (key.empty() || key == "full" || key == "debug") {
    return LogMode::kFull;
  }
  if (key == "minimal" || key == "analysis" || key == "min") {
    return LogMode::kMinimal;
  }
  return LogMode::kFull;
}

RunAction::LogMode RunAction::fLogMode =
  RunAction::ParseLogMode(ReadEnvString("DNA_LOG_MODE"));
G4bool RunAction::fNtupleMergingEnabled = true;

void RunAction::SetLogMode(const G4String& mode)
{
  const std::string key = ToLower(std::string(mode));
  const bool knownMode =
    key.empty() || key == "full" || key == "debug" ||
    key == "minimal" || key == "analysis" || key == "min";
  if (!knownMode) {
    G4cout << "RunAction: unknown log mode '" << mode
           << "'. Falling back to full." << G4endl;
  }
  fLogMode = ParseLogMode(mode);
  G4cout << "RunAction: log mode set to " << GetLogModeName() << G4endl;
}

G4String RunAction::GetLogModeName()
{
  return (fLogMode == LogMode::kMinimal) ? "minimal" : "full";
}

G4bool RunAction::IsFullLogMode()
{
  return fLogMode == LogMode::kFull;
}

G4bool RunAction::IsMinimalLogMode()
{
  return fLogMode == LogMode::kMinimal;
}

G4bool RunAction::IsStepModelDetailEnabled()
{
  return IsFullLogMode();
}

G4bool RunAction::IsTrackNtupleEnabled()
{
  return IsFullLogMode();
}

G4bool RunAction::IsNtupleMergingEnabled()
{
  return fNtupleMergingEnabled;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::RunAction() : G4UserRunAction(), fConfigNtupleId(-1), fEventNtupleId(-1)
{
  // Create analysis manager
  G4cout << "##### Create analysis manager "
         << "  " << this << G4endl;
  auto* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Register(fInelasticSum);
  accumulableManager->Register(fInelasticSqSum);
  accumulableManager->Register(fPrimaryEnergySum);
  accumulableManager->Register(fPrimaryEnergyCount);

  auto analysisManager = G4AnalysisManager::Instance();

  analysisManager->SetDefaultFileType("root");
  G4bool mergeNtuples = ReadEnvFlag("DNA_NTUPLE_MERGE", true);
  G4int nofReducedNtupleFiles = static_cast<G4int>(ReadEnvLong("DNA_NTUPLE_FILES", 0));
  if (nofReducedNtupleFiles < 0) nofReducedNtupleFiles = 0;
  const G4bool mtApp = G4Threading::IsMultithreadedApplication();
  const G4bool splitByEventsRequested = ReadEnvLong("DNA_ROOT_SPLIT_EVENTS", 0) > 0;
  const G4bool splitBySizeRequested = HasEnv("DNA_ROOT_MAX_MB");
  if (mtApp && mergeNtuples && (splitByEventsRequested || splitBySizeRequested)) {
    mergeNtuples = false;
    G4cout << "RunAction: MT ROOT rotation requested -> disabling ntuple merging "
           << "so workers can rotate files safely." << G4endl;
  }
  fNtupleMergingEnabled = mergeNtuples;
  analysisManager->SetNtupleMerging(mergeNtuples, nofReducedNtupleFiles);

  G4cout << "Using " << analysisManager->GetType() << " analysis manager" << G4endl;
  analysisManager->SetVerboseLevel(1);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

RunAction::~RunAction() {}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::ConfigureNtuples()
{
  const G4bool enableStepModelDetail = IsStepModelDetailEnabled();
  const G4bool enableStringColumns =
    enableStepModelDetail && ReadEnvFlag("DNA_NTUPLE_STRINGS", true);

  if (fNtuplesBooked) {
    if (fBookedMode != fLogMode || fEnableStringColumns != enableStringColumns) {
      G4ExceptionDescription msg;
      msg << "Log mode was changed after ntuple schema was created. "
          << "Set /dna/test/setLogMode before /run/initialize.";
      G4Exception("RunAction::ConfigureNtuples", "dna-logmode-001",
                  FatalException, msg);
    }
    return;
  }

  auto* analysisManager = G4AnalysisManager::Instance();

  analysisManager->CreateNtuple("step", "dnaphysics");
  if (enableStepModelDetail) {
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
    analysisManager->CreateNtupleDColumn("vibCrossSection");
    analysisManager->CreateNtupleIColumn("channelIndex");
    analysisManager->CreateNtupleDColumn("channelMicroXS");
    if (enableStringColumns) {
      analysisManager->CreateNtupleSColumn("processName");
      analysisManager->CreateNtupleSColumn("modelName");
    }
  } else {
    analysisManager->CreateNtupleDColumn("flagParticle");
    analysisManager->CreateNtupleDColumn("x");
    analysisManager->CreateNtupleDColumn("y");
    analysisManager->CreateNtupleDColumn("z");
    analysisManager->CreateNtupleDColumn("totalEnergyDeposit");
    analysisManager->CreateNtupleDColumn("kineticEnergy");
    analysisManager->CreateNtupleIColumn("eventID");
    analysisManager->CreateNtupleIColumn("parentID");
  }
  analysisManager->FinishNtuple();

  if (IsTrackNtupleEnabled()) {
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
  }

  fEventNtupleId = analysisManager->CreateNtuple("event", "event_energy_budget");
  analysisManager->CreateNtupleIColumn("eventID");
  analysisManager->CreateNtupleDColumn("primaryEnergy");
  analysisManager->CreateNtupleDColumn("depositedEnergy");
  analysisManager->CreateNtupleDColumn("escapedEnergy");
  analysisManager->CreateNtupleDColumn("escapedBackEnergy");
  analysisManager->CreateNtupleDColumn("escapedForwardEnergy");
  analysisManager->CreateNtupleDColumn("escapedLateralEnergy");
  analysisManager->CreateNtupleDColumn("closureEnergy");
  analysisManager->CreateNtupleDColumn("closureFraction");
  analysisManager->CreateNtupleIColumn("nEscapedTracks");
  analysisManager->CreateNtupleIColumn("nEscapedElectrons");
  analysisManager->FinishNtuple();

  fConfigNtupleId = -1;
  if (enableStepModelDetail && enableStringColumns) {
    fConfigNtupleId = analysisManager->CreateNtuple("config", "dnaphysics");
    analysisManager->CreateNtupleSColumn("key");
    analysisManager->CreateNtupleSColumn("value");
    analysisManager->FinishNtuple();
  }

  fNtuplesBooked = true;
  fEnableStringColumns = enableStringColumns;
  fBookedMode = fLogMode;

  G4cout << "DNA logging mode: " << GetLogModeName() << G4endl;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void RunAction::BeginOfRunAction(const G4Run*)
{
  auto* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Reset();

  auto analysisManager = G4AnalysisManager::Instance();
  ConfigureNtuples();

  // Open an output file
  const std::string baseNameEnv = ReadEnvString("DNA_ROOT_BASENAME");
  G4String fileName = baseNameEnv.empty() ? G4String("dna") : G4String(baseNameEnv);
  static G4bool cleaned = false;
  if (!cleaned && G4Threading::IsMasterThread()) {
    if (!ReadEnvFlag("DNA_KEEP_OLD_ROOT", false)) {
      DeleteOldRootFiles(baseNameEnv.empty() ? std::string("dna") : baseNameEnv);
    }
    cleaned = true;
  }
  analysisManager->OpenFile(fileName);

  const G4bool writeConfigHere =
    (fConfigNtupleId >= 0) &&
    (!G4Threading::IsMultithreadedApplication() || !IsMaster());
  if (writeConfigHere) {
    const std::string physRawEnv = ReadEnvString("DNA_PHYSICS");
    const std::string physRaw = physRawEnv.empty() ? "ice_hex" : physRawEnv;
    const std::string physLower = ToLower(physRaw);
    const std::string icePhase = NormalizeIcePhase(physLower);
    const std::string physChoice = (physLower == "water") ? "water" : "ice";

    auto appendConfig = [&](const std::string& key, const std::string& value) {
      analysisManager->FillNtupleSColumn(fConfigNtupleId, 0, key);
      analysisManager->FillNtupleSColumn(fConfigNtupleId, 1, value);
      analysisManager->AddNtupleRow(fConfigNtupleId);
    };

    appendConfig("DNA_PHYSICS", physRaw);
    appendConfig("physics_list", physChoice);
    appendConfig("ice_phase", icePhase.empty() ? "n/a" : icePhase);
    appendConfig("log_mode", GetLogModeName());
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

  // Merge accumulables from worker threads before reporting.
  auto* accumulableManager = G4AccumulableManager::Instance();
  accumulableManager->Merge();

  // Print histogram statistics
  auto analysisManager = G4AnalysisManager::Instance();

  const G4bool writeConfigHere =
    (fConfigNtupleId >= 0) &&
    (!G4Threading::IsMultithreadedApplication() || !IsMaster());
  if (writeConfigHere) {
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

  PrintWValueSummary(aRun);

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

void RunAction::AccumulateEventIonisations(G4double nInelastic)
{
  if (nInelastic < 0.) return;
  fInelasticSum += nInelastic;
  fInelasticSqSum += nInelastic * nInelastic;
}

void RunAction::AccumulatePrimaryEnergy(G4double energy)
{
  if (energy <= 0.) return;
  fPrimaryEnergySum += energy;
  fPrimaryEnergyCount += 1.;
}

void RunAction::PrintWValueSummary(const G4Run* aRun)
{
  if (!IsMaster()) return;

  const G4int nofEvents = aRun ? aRun->GetNumberOfEvent() : 0;
  if (nofEvents <= 0) return;

  const G4double meanInelastic = fInelasticSum.GetValue() / nofEvents;
  const G4double meanInelastic2 = fInelasticSqSum.GetValue() / nofEvents;
  G4double rms = meanInelastic2 - meanInelastic * meanInelastic;
  rms = (rms > 0.) ? std::sqrt(rms) : 0.;

  G4double primaryEnergy = 0.;
  if (fPrimaryEnergyCount.GetValue() > 0.) {
    primaryEnergy = fPrimaryEnergySum.GetValue() / fPrimaryEnergyCount.GetValue();
  }

  std::ios::fmtflags oldFlags = G4cout.flags();
  std::streamsize oldPrecision = G4cout.precision();
  G4cout.setf(std::ios::fixed, std::ios::floatfield);
  G4cout.precision(3);

  G4cout << "\n Nb of ionisations = " << meanInelastic << " +- " << rms << G4endl;
  if (meanInelastic > 0. && primaryEnergy > 0.) {
    const G4double w = primaryEnergy / meanInelastic;
    const G4double wErr = primaryEnergy * rms / (meanInelastic * meanInelastic);
    G4cout << "\n w = " << G4BestUnit(w, "Energy") << " +- "
           << G4BestUnit(wErr, "Energy") << G4endl;

    std::string outPath = ReadEnvString("DNA_WVALUE_FILE");
    if (outPath.empty()) outPath = "wvalue_ice.txt";
    std::ofstream out(outPath, std::ios::app);
    if (out.good()) {
      out << std::scientific << std::setprecision(6)
          << primaryEnergy / eV << ' '
          << meanInelastic << ' '
          << rms << ' '
          << w / eV << ' '
          << wErr / eV
          << '\n';
    } else {
      G4cout << "RunAction: failed to append W-value output file '" << outPath << "'." << G4endl;
    }
  } else {
    G4cout << "\n w = undefined (need positive primary energy and ionisation mean)." << G4endl;
  }

  G4cout.flags(oldFlags);
  G4cout.precision(oldPrecision);
}
