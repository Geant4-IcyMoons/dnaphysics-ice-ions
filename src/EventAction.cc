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
/// \file EventAction.cc
/// \brief Implementation of the EventAction class

#include "EventAction.hh"

#include "G4AnalysisManager.hh"
#include "G4Event.hh"
#include "G4SystemOfUnits.hh"
#include "G4ios.hh"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

namespace {
G4long ReadEnvLong(const char* name, G4long defaultValue)
{
  const char* env = std::getenv(name);
  if (!env || !*env) return defaultValue;
  char* end = nullptr;
  const long val = std::strtol(env, &end, 10);
  if (end == env) return defaultValue;
  return static_cast<G4long>(val);
}

G4String TrimRootExt(const G4String& value)
{
  if (value.size() >= 5 && value.substr(value.size() - 5) == ".root") {
    return value.substr(0, value.size() - 5);
  }
  return value;
}

bool FileExists(const std::string& path)
{
  std::error_code ec;
  return std::filesystem::exists(path, ec);
}
}

EventAction::EventAction()
{
  const G4long splitEvents = ReadEnvLong("DNA_ROOT_SPLIT_EVENTS", 10);
  if (splitEvents > 0) {
    fSplitEveryEvents = static_cast<G4int>(splitEvents);
  }
  const G4long maxMb = ReadEnvLong("DNA_ROOT_MAX_MB", 1024);
  fMaxBytes = maxMb * 1024L * 1024L;
  fCheckEvery = static_cast<G4int>(ReadEnvLong("DNA_ROOT_CHECK_EVERY", 1000));
  if (fCheckEvery < 1) fCheckEvery = 1;
}

void EventAction::EndOfEventAction(const G4Event* event)
{
  if (!event) return;
  const G4int eventId = event->GetEventID();
  if (fSplitEveryEvents > 0) {
    if (((eventId + 1) % fSplitEveryEvents) == 0) {
      RotateFile();
    }
    return;
  }
  if ((eventId % fCheckEvery) != 0) return;
  MaybeRotate();
}

void EventAction::MaybeRotate()
{
  auto* analysisManager = G4AnalysisManager::Instance();
  if (!analysisManager || !analysisManager->IsOpenFile()) return;

  if (!fInitialized) {
    fBaseName = TrimRootExt(analysisManager->GetFileName());
    if (fBaseName.empty()) fBaseName = "dna";
    fInitialized = true;
  }

  // Force a write so the on-disk size is up to date before checking.
  std::string path = CurrentFilePath();
  if (!FileExists(path)) return;

  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec || size < static_cast<std::uintmax_t>(fMaxBytes)) return;

  RotateFile();
}

std::string EventAction::CurrentFilePath() const
{
  const G4String base = BuildFileBaseName();
  std::string candidate = base;
  if (FileExists(candidate)) return candidate;
  if (candidate.size() < 5 || candidate.substr(candidate.size() - 5) != ".root") {
    candidate += ".root";
  }
  return candidate;
}

G4String EventAction::BuildFileBaseName() const
{
  if (fFileIndex == 0) return fBaseName;
  std::ostringstream name;
  name << fBaseName << "_" << std::setw(3) << std::setfill('0') << fFileIndex;
  return name.str();
}

void EventAction::RotateFile()
{
  auto* analysisManager = G4AnalysisManager::Instance();
  if (!analysisManager || !analysisManager->IsOpenFile()) return;

  if (!fInitialized) {
    fBaseName = TrimRootExt(analysisManager->GetFileName());
    if (fBaseName.empty()) fBaseName = "dna";
    fInitialized = true;
  }

  analysisManager->Write();
  analysisManager->CloseFile(false);

  ++fFileIndex;
  const G4String nextBase = BuildFileBaseName();
  analysisManager->OpenFile(nextBase);

  G4cout << "Rotated ROOT output to " << nextBase << ".root" << G4endl;
}
